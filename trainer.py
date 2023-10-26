from pathlib import Path
import re
import os
from shutil import rmtree

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from typing import Optional
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from utautai.soundstorm import SoundStorm
from utautai.dataset.data_processor import DataProcessor
from utautai.optimizer import get_optimizer
from utautai.prompt_tts.style_module import StyleModule
import joblib

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


class UTAUTAI_Trainer(nn.Moduele):
    def __init__(
        self,
        model: SoundStorm,
        kmeans, 
        dataprocessor: DataProcessor,
        stylemodule: StyleModule,
        *,
        num_warmup_steps,
        batch_size,
        epochs = 20,
        is_raw_wav: bool = False,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        log_steps = 10,
        save_model_steps = 5000,
        results_folder = './results',
        log_dir = './log',
        accelerate_kwargs: dict = dict(),
        split_batches = False,
        num_ckpt_keep = 8,
        use_tensorboard = False,
        loss_weight = [1, 1]
    ):
        
        super().__init__()
        
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        else:
            self.tb_writer = None
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches = split_batches,
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs
        )

        self.model = model
        self.register_buffer('steps', torch.Tensor([0]))

        self.epochs = epochs
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        
        # max grad norm
        self.max_grad_norm = max_grad_norm
        
        #create dataset
        self.data_processor = dataprocessor
        self.dl = self.data_processor.train_loader
        self.valid_dl = self.data_processor.valid_loader
        
        #create style module
        self.style_module = stylemodule
        self.kmeans = kmeans #mert_kmeans for infer
        self.loss_weight = loss_weight
        if self.loss_weight is not None:
            self.lambda1 = loss_weight[0] # for soundstorm
            self.lambda2 = loss_weight[1] # for stylemodule
        
        if self.is_main:
            self.data_processor.print_stats()
        
        self.is_raw_wav = is_raw_wav
        
        #optimizer
        self.optim = get_optimizer(model.parameters(), lr=lr, wd=wd)
        
        
        #lr and scheduler
        self.lr = lr
        self.initial_lr = initial_lr
        num_train_steps = epochs * self.data_processor.train_dataset.__len__() // (batch_size * grad_accum_every)
        self.scheduler = CosineAnnealingLR(self.optim, T_max=num_train_steps)
        
        #prepare with accelerator
        
        (
            self.model, 
            self.style_module,
            self.optim, 
            self.scheduler, 
            self.dl, 
            self.valid_dl
        ) = self.accelerator.prepare(
                self.model, 
                self.style_module,
                self.optim,
                self.scheduler, 
                self.dl,
                self.valid_dl
        ) 
        
        #datloader iterators
        
        self.log_steps = log_steps
        self.save_model_steps = save_model_steps
        
        self.results_folder = Path(results_folder)
        self.num_ckpt_keep = num_ckpt_keep
        
        if not results_folder.exists():
            self.results_folder.mkdir(parents=True, exist_ok=True)
            
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": num_warmup_steps, "learning_rate": lr, "initial_learning_rate": lr, "epochs": epochs}
        self.accelerator.init_trackers("soundstorm", config=hps)
        self.best_dev_loss = float('inf')
        
    def save(self, path, stylemodule_path, dev_loss):
        if dev_loss < self.best_dev_loss:
            self.best_dev_loss = dev_loss
            torch.save(self.accelerator.get_state_dict(self.model), f'{self.results_folder}/SoundStorm_best_dev.pt')
            torch.save(self.accelerator.get_state_dict(self.style_module), f'{self.results_folder}/StyleModule_best_dev.pt')        
        ckpts = sorted(Path(path).parent.glob(f'SoundStormTrainer_*'))
        stylemodule_ckpts = sorted(Path(path).parent.glob(f'StyleModuleTrainer_*'))
        if len(ckpts) > self.num_ckpt_keep:
            [os.remove(c) for c in ckpts[:-self.num_ckpt_keep]]
            [os.remove(c) for c in stylemodule_ckpts[:-self.num_ckpt_keep]]
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict(),
            best_dev_loss = self.best_dev_loss
        )
        stylemodule_pkg = dict(
            model=self.accelerator.get_state_dict(self.style_module),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict(),
            best_dev_loss = self.best_dev_loss
        )
        torch.save(pkg, path)
        torch.save(stylemodule_pkg, stylemodule_path)
        
    def load(self, path = None, stylemodule_path = None, restore_optimizer = True):
        if not exists(path):
            ckpts = sorted(self.results_folder.glob(f'SoundStormTrainer_*'))
            path = str(ckpts[-1])
        if not exists(stylemodule_path):
            stylemodule_ckpts = sorted(self.results_folder.glob(f'StyleModuleTrainer_*'))
            stylemodule_path = str(ckpts[-1])
            
        model = self.accelerator.unwrap_model(self.model)
        pkg = torch.load(path, map_location='cpu')
        model.load_state_dict(pkg['model'])
        
        stylemodule = self.accelerator.unwrap_model(self.style_module)
        stylemodule_pkg = torch.load(stylemodule_path, map_location='cpu')
        stylemodule.load_state_dict(stylemodule_pkg['model'])

        if restore_optimizer:
            self.optim.load_state_dict(pkg['optim'])
            self.scheduler.load_state_dict(pkg['scheduler'])
            if 'best_dev_loss' in pkg.keys():
                self.best_dev_loss = pkg['best_dev_loss']
                if self.is_main:
                    self.print(f'The best dev loss before is {self.best_dev_loss}')

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)
            
    def print(self, msg):
        self.accelerator.print(msg)
    
    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr
        
    def train(self):
        self.model.train()
        
        grad_accum = 0
        logs = {}
        steps = int(self.steps.item())
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
        
        for epoch in range(self.epochs):
            if self.is_main:
                print(f'Epoch:{epoch} start...')
            
            for batch in self.dl:
                (acoustic_token_ids, music_semantic_token_ids, 
                 lyrics_semantic_token_ids, mert_feat) = (batch['audio_features'],
                                              batch['stok_music'],
                                              batch['stok_lyric'],
                                              batch['mert_feat'])
                prompts = batch['prompts']
                variation_loss, _, _ = self.style_module(prompts, mert_feat)
                semantic_token_ids = torch.cat([lyrics_semantic_token_ids, music_semantic_token_ids],
                                               axis=1)
                
                loss, acc, _ = self.model(x=acoustic_token_ids, cond_ids=semantic_token_ids)
                
                if self.loss_weight is not None:
                    all_loss = loss * self.lambda1 + variation_loss * self.lambda2
                else:
                    all_loss = loss + variation_loss # loss weighting should be considered.
                
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('Loss/train', loss.item(), steps)
                    self.tb_writer.add_scalar('Accuracy/train', acc.item(), steps)
                    self.tb_writer.add_scalar('Variation_Loss/train', variation_loss, steps)
                    self.tb_writer.add_scalar('All_Loss/train', all_loss, steps)
                    
                accum_log(logs, {'loss': loss.item() / self.grad_accum_every, 'acc': acc.item() / self.grad_accum_every,
                                 'variation_loss': variation_loss.item()/ self.grad_accum_every,
                                 'all_loss': all_loss.item() / self.grad_accum_every})
                
                self.accelerator.backward(all_loss/ self.grad_accum_every)
                grad_accum += 1
                
                # update params
                if grad_accum == self.grad_accum_every:
                        if exists(self.max_grad_norm):
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optim.step()
                        self.optim.zero_grad()
                        grad_accum = 0
                        
                        # log
                        if self.is_main and not (steps % self.log_steps):
                            self.print(f"Epoch {epoch} -- Step {steps}: loss: {logs['loss']:0.3f}\tacc:{logs['acc']:0.3f}\tvariation_loss:{logs['variation_loss']:0.3f}\tall_loss:{logs['all_loss']:0.3f}")
                            self.accelerator.log({"train/loss": logs['loss'], "train/acc": logs['acc'], "train/variation_loss":logs['variation_loss'], "train/all_loss":logs['all_loss'], "train/learning_rate": lr}, step=steps)
                        logs = {}
                        
                        self.accelerator.wait_for_everyone()
                        
                        # validate and save model
                        if self.is_main and not(steps % self.save_model_steps):
                            
                            # validate
                            losses = []
                            total_loss = 0.0
                            total_acc = 0.0
                            num = 0
                            self.model.eval()
                            for batch in self.valid_dl:
                                with torch.inference_mode():
                                    (acoustic_token_ids, music_semantic_token_ids, 
                                     lyric_semantic_token_ids, mert_feat) = (batch['audio_features'],
                                              batch['stok_music'],
                                              batch['stok_lyric'],
                                              batch['mert_feat'])
                                    prompts = batch['prompts']
                                    variation_loss, _, _ = self.style_module(prompts, mert_feat)
                                    semantic_token_ids = torch.cat([lyric_semantic_token_ids, music_semantic_token_ids], axis=1)
                                    b = semantic_token_ids.size(0)
                                    num += b
                                    
                                    loss, acc, _ = self.model(x = acoustic_token_ids, cond_ids=semantic_token_ids)
                                    if self.loss_weight is not None:
                                        all_loss = loss * self.lambda1 + variation_loss * self.lambda2
                                    else:
                                        all_loss = loss + variation_loss # loss weighting should be considered.
                                    total_loss += all_loss.item() * b
                                    losses.append(all_loss.item())
                                    total_acc += acc.item() * b
                            self.print(f'{steps}: valid loss {total_loss / num:0.3f}, valid acc {total_acc / num:0.3f}')  
                            self.accelerator.log({"valid/loss": total_loss / num, "valid/acc": total_acc / num}, step=steps) 
                            if self.tb_writer is not None:
                                self.tb_writer.add_scalar('Loss/validate', total_loss / num, steps)
                                self.tb_writer.add_scalar('Accuracy/validate', total_acc / num, steps)
                            
                            # save model
                            model_path = str(self.results_folder / f'SoundStormTrainer_{steps:08d}')
                            stylemodule_path = str(self.results_folder / f'StyleModuleTrainer_{steps:08d}')
                            self.save(model_path, stylemodule_path, total_loss / num)                        
                            self.print(f'{steps}: saving model to {str(self.results_folder)}')
                            self.model.train()
                            
                        # Update lr    
                        self.steps += 1
                        steps = int(self.steps.item())               
                        if steps < self.num_warmup_steps:
                            lr = self.warmup(steps)
                            for param_group in self.optim.param_groups:
                                param_group['lr'] = lr
                        else:
                            self.scheduler.step() 
                            lr = self.scheduler.get_last_lr()[0]       
            
        self.print('training complete')
        
    def continue_train(self):
        self.load()
        self.train()