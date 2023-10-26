import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from .files_dataset import FilesAudioDataset
from .collate import collate
from .sampler import DynamicBatchSampler

class OffsetDataset(Dataset):
    def __init__(self, dataset, start, end, test=False):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        self.test = test
        assert 0 <= self.start < self.end <= len(self.dataset)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.dataset.get_item(self.start + item, test=self.test)

class DataProcessor():
    def __init__(self, batch_size: int, num_workers: int, 
                 dataset_dir: str, sr: int, channels: int, train_test_split: float,  
                 min_duration: int, max_duration: int, num_buckets: int,  
                 sample_length: int, aug_shift: bool, cache_dir: str, labels: bool, 
                 device: str, n_tokens: int, train_semantic: bool):
        self.dataset = FilesAudioDataset(sr=sr, channels=channels, min_duration=min_duration,
                                         max_duration=max_duration, sample_length=sample_length,
                                         cache_dir=cache_dir, aug_shift=aug_shift, labels=labels,
                                         device=device, dataset_dir=dataset_dir, n_tokens=n_tokens, 
                                         train_semantic=train_semantic)
        self.create_datasets(train_test_split=train_test_split)
        self.create_samplers(num_buckets=num_buckets, max_duration=max_duration)
        self.create_data_loaders(batch_size=batch_size, num_workers=num_workers, labels=labels)
        self.print_stats()

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.valid_sampler.set_epoch(epoch)

    def create_datasets(self, train_test_split):
        train_len = int(len(self.dataset) * train_test_split)
        self.train_dataset = OffsetDataset(self.dataset, 0, train_len, test=False)
        self.valid_dataset = OffsetDataset(self.dataset, train_len, len(self.dataset), test=True)

    def create_samplers(self, num_buckets=10, max_duration=120):
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_sampler = DynamicBatchSampler(train_sampler, self.train_dataset.get_dur,
                                                   num_bukets=num_buckets,
                                                   max_size=20, max_tokens=max_duration)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset)
        self.valid_sampler = DynamicBatchSampler(valid_sampler, self.valid_dataset.get_dur,
                                                num_buckets=num_buckets,
                                                max_size=20, max_tokens=max_duration)

    def create_data_loaders(self, batch_size, num_workers, labels):
        # Loader to load mini-batches
        if labels:
            collate_fn = collate

        logging.info('Creating Data Loader')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                       num_workers=num_workers, sampler=self.train_sampler, 
                                       pin_memory=False, drop_last=True, collate_fn=collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, 
                                       num_workers=num_workers, sampler=self.valid_sampler, 
                                       pin_memory=False, drop_last=False, collate_fn=collate_fn)

    def print_stats(self):
        logging.info(f"Train {len(self.train_dataset)} samples. Test {len(self.test_dataset)} samples")
        logging.info(f'Train sampler: {self.train_sampler}')
        logging.info(f'Train loader: {len(self.train_loader)}')