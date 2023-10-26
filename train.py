import hydra
import omegaconf
import joblib

from trainer import UTAUTAI_Trainer
from utautai.soundstorm import SoundStorm, ConformerWrapper
from utautai.prompt_tts.style_module import StyleModule
from utautai.dataset.data_processor import DataProcessor

def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct

@hydra.main(config_name='config', version_base=None, config_path='configs')
def main(cfg: omegaconf.DictConfig) -> None:
    conformer_kwargs = cfg.conformerwrapper
    conformer = ConformerWrapper(codebook_size=conformer_kwargs.codebook_size,
                                 num_quantizers=conformer_kwargs.num_quantizers,
                                 conformer=dict_from_config(conformer_kwargs.conformer))
    
    soundstorm_kwargs = cfg.soundstorm
    soundstorm = SoundStorm(net=conformer,
                            num_semantic_token_ids=soundstorm_kwargs.num_semantic_token_ids,
                            semantic_pad_id=soundstorm_kwargs.semantic_pad_id,
                            pad_id=soundstorm_kwargs.pad_id,
                            schedule=soundstorm_kwargs.schedule)
    
    stylemodule_kwargs = cfg.stylemodule
    stylemodule = StyleModule(name=stylemodule_kwargs.bert_name,
                              output_dim=stylemodule_kwargs.output_dim,
                              prompt_seq_len=stylemodule_kwargs.prompt_seq_len,
                              mert_seq_len=stylemodule_kwargs.mert_seq_len,
                              dim_head=stylemodule_kwargs.dim_head,
                              heads=stylemodule_kwargs.heads,
                              depth=stylemodule_kwargs.depth,
                              device=stylemodule_kwargs.device,
                              timesteps=stylemodule_kwargs.timesteps,
                              causal=stylemodule_kwargs.causal,
                              use_ddim=stylemodule_kwargs.use_ddim,
                              noise_schedule=stylemodule_kwargs.noise_schedule,
                              objective=stylemodule_kwargs.objective)
    
    kmeans_kwargs = cfg.kmeans
    kmeans = joblib.load(kmeans_kwargs.path)
    
    dataprocessor_kwargs = cfg.dataprocessor
    dataprocessor = DataProcessor(sr=dataprocessor_kwargs.sr,
                                  channels=dataprocessor_kwargs.channels,
                                  batch_size=dataprocessor_kwargs.batch_size,
                                  num_workers=dataprocessor_kwargs.num_workers,
                                  dataset_dir=dataprocessor_kwargs.dataset_dir,
                                  train_test_split=dataprocessor_kwargs.train_test_split,
                                  min_duration=dataprocessor_kwargs.min_duration,
                                  max_duration=dataprocessor_kwargs.max_duration,
                                  num_buckets=dataprocessor_kwargs.num_buckets,
                                  sample_length=dataprocessor_kwargs.sample_length,
                                  aug_shift=dataprocessor_kwargs.aug_shift,
                                  cache_dir=dataprocessor_kwargs.cache_dir,
                                  labels=dataprocessor_kwargs.labels,
                                  device=dataprocessor_kwargs.device,
                                  n_tokens=dataprocessor_kwargs.n_tokens,
                                  train_semantic=dataprocessor_kwargs.train_semantic)
    
    trainer_kwargs = cfg.trainer
    trainer = UTAUTAI_Trainer(model=soundstorm, 
                              stylemodule=stylemodule,
                              kmeans=kmeans, 
                              dataprocessor=dataprocessor, 
                              num_warmup_steps=trainer_kwargs.num_warmup_steps,
                              batch_size=trainer_kwargs.batch_size,
                              epochs=trainer_kwargs.epochs, 
                              lr=trainer_kwargs.lr,
                              initial_lr=trainer_kwargs.initial_lr,
                              grad_accum_every=trainer_kwargs.grad_accum_every,
                              log_steps=trainer_kwargs.log_steps,
                              save_model_steps=trainer_kwargs.save_model_steps,
                              results_folder=trainer_kwargs.results_folder,
                              log_dir=trainer_kwargs.log_dir,
                              accelerate_kwargs=dict_from_config(trainer_kwargs.accelerate_kwargs),
                              num_ckpt_keep=trainer_kwargs.num_ckpt_keep,
                              use_tensorboard=trainer_kwargs.use_tensorboard,
                              loss_weight=trainer_kwargs.loss_weight)
    trainer.train()
    
    
if __name__ == '__main__':
    main()