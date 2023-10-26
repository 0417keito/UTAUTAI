import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
import os
import logging
import random
import omegaconf
import hydra
import warnings
from pathlib import Path
from shutil import copyfile
from t2s_utils.opt_and_scheduler import get_optimizer, get_scheduler, Eden
from t2s_utils.utils import (AttributeDict, MetricsTracker, get_params, set_batch_count,
                         display_and_save_batch, save_checkpoint_with_global_batch_idx,
                         remove_checkpoints, save_checkpoint_impl, fix_random_seed)
from prompt_tts.style_module import get_style_module
from dataset.data_processor import DataProcessor

class T2S_Trainer(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir = cfg.log_dir
            )
        else: 
            self.tb_writer = None
        
        device = cfg.device
        logging.info(f"Device: {device}")
        if cfg.style_module is not None:
            self.use_condition = True
            self.style_module = get_style_module(cfg.style_module)
        else: self.use_condition = False
        
        logging.info("Create Model")
        if cfg.resume_from:
            self.model = self.resume_from(device)
        else:            
            self.model = self.get_model(device)
        num_params = sum([p.numel() for p in self.model.parameters()])
        logging.info(f"Number of model parameters: {num_params}")
        
        self.optimizer = get_optimizer(self.model.parameters(), lr=cfg.base_lr)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.optimizer.zero_grad()
        
        self.data_processor = DataProcessor(cfg)
        self.train_dl = self.data_processor.train_loader
        self.valid_dl = self.data_processor.valid_loader
        
        self.scaler = GradScaler(
            enabled=(cfg.dtype in ["fp16", "float16"]), init_scale=1.0
        )
        
    def get_model(self, cfg, device):
        from t2s_models.valle import VALLE
        model_cfg = cfg.model
        model = VALLE(
            model_cfg.n_dim, model_cfg.num_head, model_cfg.num_layers,
            norm_first=True, add_prenet=False, prefix_mode=model_cfg.prefix_mode,
            share_embedding=True, nar_scale_factor=1.0, prepend_bos=True,
            num_quantizers=model_cfg.num_quantizers
        ).to(device)
        self.start_epoch = 0
        
        return model
        
    def resume_from(self, cfg, device): 
        model = self.get_model(self, cfg, device)
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True)
        assert not missing_keys
        self.start_epoch = checkpoint["start_epoch"] + checkpoint["num_epochs"]
        
        return model
    
    def save_checkpoint(self, training_dict:AttributeDict, exp_dir):
        filename = exp_dir/f"epoch-{training_dict.cur_epoch}.pt"
        save_checkpoint_impl(
            filename=filename,
            model=self.model,
            training_dict=training_dict,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=None,
            scaler=self.scaler,
        )
        if training_dict.best_train_epoch == training_dict.cur_epoch:
            best_train_filename = exp_dir/"best-train-loss.pt"
            copyfile(src=filename, dst=best_train_filename)
        if training_dict.best_valid_epoch == training_dict.cur_epoch:
            best_valid_filename = exp_dir/"best-valid-loss.pt"
            copyfile(src=filename, dst=best_valid_filename)
        
    def compute_loss(self,
                     cfg: omegaconf.DictConfig,
                     training_dict: AttributeDict,
                     batch: dict,
                     is_training: bool,
                     is_condition: bool = False):
        
        device = cfg.device
        variation_loss, prompt_rep, ref_rep = self.style_module(batch)
        conditions = torch.cat((prompt_rep, ref_rep), dim=1)
        text_tokens = batch["text_tokens"].to(device)
        text_tokens_lens = batch["text_tokens_lens"].to(device)
        assert text_tokens.ndim == 2
        
        audio_features = batch["audio_features"].to(device)
        audio_features_lens = batch["audio_features_lens"].to(device)
        assert audio_features.ndim == 3
        
        with torch.set_grad_enabled(is_training):
            predicts, loss, metrics = self.model(
                x=text_tokens,
                x_lens=text_tokens_lens,
                y=audio_features,
                y_lens=audio_features_lens,
                train_stage=cfg.train_stage,
                conditions=conditions
            )
        
        assert loss.requires_grad == is_training
        
        info = MetricsTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info["frames"] = (audio_features_lens).sum().item()
            info["utterances"] = text_tokens.size(0)
        
        info["loss"] = loss.detach().cpu().item()
        for metric in metrics:
            info[metric] = metrics[metric].detach().cpu().item()
        del metrics
        
        return predicts, loss, info   
    
    def compute_validation_loss(
        self,
        cfg: omegaconf.DictConfig,
        training_dict: AttributeDict,
        valid_dl: torch.utils.data.DataLoader,
    ):
        for batch_idx, batch in enumerate(valid_dl):
            predicts, loss, loss_info = self.compute_loss(
                cfg=cfg,
                training_dict=training_dict,
                model=self.model,
                batch=batch,
                is_training=False,
            )
            assert loss.requires_grad is False
            tot_loss = tot_loss + loss_info
        loss_value = tot_loss["loss"] / tot_loss["frames"]
        if loss_value < training_dict.best_valid_loss:
            training_dict.best_valid_epoch = training_dict.cur_epoch
            training_dict.best_valid_loss = loss_value
            
        if cfg.visualize:
            output_dir = Path(
                f"{cfg.exp_dir}/eval/step-{training_dict.batch_idx_train:06d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self.model.visualize(predicts, batch, output_dir=output_dir)
        
        return tot_loss
    
    def train_one_epoch(
        self,
        cfg: omegaconf.DictConfig,
        rng: random.Random,
        training_dict: AttributeDict,
        use_condition: bool,
    ):
        self.model.train()
        tot_loss = MetricsTracker()
        iter_dl = iter(self.train_dl)
        dtype, enabled = torch.float32, False
        if cfg.dtype in ["bfloat16", "bf16"]:
            dtype, enabled = torch.bfloat16, True
        elif cfg.dtype in ["float16", "fp16"]:
            dtype, enabled = torch.float16, True
        
        batch_idx = 0
        while True:
            try:
                batch = next(iter_dl)
            except StopIteration:
                logging.info("Reaches end of dataloader.")
                break
            
            batch_idx += 1
            training_dict.batch_idx_train += 1
            batch_size = len(batch["text"])
                
            try:
                with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                    _, loss, loss_info = self.compute_loss(
                        cfg=cfg, training_dict=training_dict,
                        model=self.model, batch=batch, is_training=True
                    )
                
                tot_loss = (
                    tot_loss * (1 - 1 / training_dict.reset_interval)
                ) + loss_info * (1 / training_dict.reset_interval)
                
                self.scaler(loss).backward()
                if training_dict.batch_idx_train >= cfg.accumulate_grad_steps:
                    if (
                        training_dict.batch_idx_train % cfg.accumulate_grad_steps
                        == 0
                    ):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        for k in range(cfg.accumulate_grad_steps):
                            if isinstance(self.scheduler, Eden):
                                self.scheduler.step_batch(training_dict.batch_idx_train)
                            else:
                                self.scheduler.step()
                
                set_batch_count(self.model, training_dict.batch_idx_train)
            except:
                display_and_save_batch(batch, cfg.exp_dir)
                raise
            
            if (training_dict.batch_idx_train > 0
                and training_dict.batch_idx_train % cfg.save_every_n == 0):
                
                save_checkpoint_with_global_batch_idx(
                    out_dir=cfg.exp_dir,
                    global_batch_idx=training_dict.batch_idx_train,
                    model=self.model,
                    training_dict=training_dict,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    sampler=None,
                    scaler=self.scaler,
                )
                
                remove_checkpoints(
                    out_dir=cfg.exp_dir,
                    topk=cfg.keep_last_k,
                )
            
            if batch_idx % 100 == 0 and cfg.dtype in ["float16", "fp16"]:
                cur_grad_scale = self.scaler._scale.item()
                if cur_grad_scale < 1.0 or (
                        cur_grad_scale < 8.0 and batch_idx % 400 == 0
                ):
                    self.scaler.update(cur_grad_scale * 2.0)
                
                if cur_grad_scale < 0.01:
                    logging.warning(f"Grad scale is small: {cur_grad_scale}")
                if cur_grad_scale < 1.0e-05:
                    raise RuntimeError(
                        f"grad_scale is too small, exiting: {cur_grad_scale}"
                    )
            
            if batch_idx % training_dict.log_interval == 0:
                cur_lr = self.scheduler.get_last_lr()[0]
                cur_grad_scale = (
                    self.scaler._scale.item()
                    if cfg.dtype in ["float16", "fp16"]
                    else 1.0
                )
                
                logging.info(
                    f"Epoch {training_dict.cur_epoch},"
                    f"batch {batch_idx}, train_loss[{loss_info}],"
                    f"tot_loss[{tot_loss}],"
                    f"batch size: {batch_size},"
                    f"lr: {cur_lr: .2e}"
                    + (
                        f", grad_scale: {cur_grad_scale}"
                        if cfg.dtype in ["float16", "fp16"]
                        else ""
                    )
                )
                
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        "train/learning_rate", cur_lr, training_dict.batch_idx_train
                    )
                    
                    loss_info.write_summary(
                        self.tb_writer,
                        "train/current_",
                        training_dict.batch_idx_train,
                    )
                    tot_loss.write_summary(
                        self.tb_writer, "train/tot_", training_dict.batch_idx_train
                    )
                    if cfg.dtype in ["float16", "fp16"]:
                        self.tb_writer.add_scalar(
                            "train/grad_scale",
                            cur_grad_scale,
                            training_dict.batch_idx_train,
                        )
            
            if training_dict.batch_idx_train % training_dict.valid_interval == 0:
                self.model.eval()
                logging.info("Computing validation loss")
                with torch.cuda.amp.autocast(dtype=dtype):
                    valid_info = self.compute_validation_loss(
                        cfg=cfg, training_dict=training_dict, valid_dl=self.valid_dl
                    )
                logging.info(
                    f"Epoch {training_dict.cur_epoch}, validation: {valid_info}"
                )
                logging.info(
                    f"Maximum memory allcated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
                )
                
                if self.tb_writer is not None:
                    valid_info.write_summary(
                        self.tb_writer, "train/valid_", training_dict.batch_idx_train
                    )
                self.model.train()
        
        loss_value = tot_loss["loss"] / tot_loss["frames"]
        training_dict.train_loss = loss_value
        if loss_value < training_dict.best_train_loss:
            training_dict.best_train_epoch = training_dict.cur_epoch
            training_dict.best_train_loss = training_dict.train_loss
        
        training_dict.cur_epoch += 1
            
            
    def train(self):
        training_dict = get_params()
        fix_random_seed(self.cfg.seed)
        rng = random.Random(self.cfg.seed)
        for epoch in range(self.start_epoch, self.cfg.num_epochs + 1):
            if isinstance(self.scheduler, Eden):
                self.scheduler.step_epoch(epoch - 1)
            
            self.data_processor.set_epoch(epoch)
            fix_random_seed(self.cfg.seed + epoch - 1)
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/epoch", epoch, training_dict.batch_idx_train)
            
            self.train_one_epoch(
                cfg=self.cfg, rng=rng, training_dict=training_dict, use_condition=self.use_condition)
            self.save_checkpoint(training_dict=training_dict, exp_dir=self.cfg.exp_dir)
            
            logging.info("Done!")
            
@hydra.main(config_name="config", config_path="./configs")
def main(cfg: omegaconf.DictConfig):
    trainer = T2S_Trainer(cfg)
    trainer.train()
    
if __name__ == "__main__":
    main()