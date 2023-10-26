import logging
import librosa
import math
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .labels import Labeller
from .utils import get_duration_sec, load_audio
from .audio_processor import AudioTokenizer, tokenize_audio


class FilesAudioDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, sample_length, cache_dir,
                 aug_shift, labels, device, dataset_dir, n_tokens, train_semantic):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr)
        self.max_duration = max_duration or math.inf
        self.sample_length = sample_length
        assert sample_length / sr < self.min_duration, f'Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = aug_shift
        self.labels = labels
        self.device = device
        self.audio_files_dir = f'{dataset_dir}/audios'
        self.train_semantic = train_semantic
        self.mert_unit_dir = f'{dataset_dir}/mert_unit'
        self.mert_dir = f'{dataset_dir}/mert'
        if self.labels:
            self.lyrics_files_dir = f'{dataset_dir}/lyrics'
            self.lyrics_alignment_dir = f'{dataset_dir}/alignment'
            genres_path = f'{dataset_dir}/id_genres.csv'
            information_path = f'{dataset_dir}/id_information.csv'
            lang_path = f'{dataset_dir}/id_lang.csv'
            metadata_path = f'{dataset_dir}/id_metadata.csv'
            tags_path = f'{dataset_dir}/id_tags.csv'
            
            genres_df = pd.read_csv(genres_path)
            self.id_to_genres = genres_df.set_index('id')['genres'].to_dict()
            information_df = pd.read_csv(information_path)
            self.id_to_info = information_df.set_index('id').to_dict('index')
            lang_df = pd.read_csv(lang_path)
            self.id_to_lang = lang_df.set_index('id')['lang'].to_dict()
            metadata_df = pd.read_csv(metadata_path)
            self.id_to_metadata = metadata_df.set_index('id').to_dict('index')
            tags_df = pd.read_csv(tags_path)
            self.id_to_tags = tags_df.set_index('id')['tags'].to_dict()
            
        self.init_dataset(n_tokens, sample_length, cache_dir)

    def filter(self, files, durations):
        # Remove files too short or too long[]
        keep = []
        for i in range(len(files)):
            filepath = files[i]
            song_id = os.path.splitext(os.path.basename(filepath))[0]
            lang = self.id_to_lang[song_id]
            if lang != 'en':
                continue
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        logging.info(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        logging.info(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep] #サンプル長
        self.cumsum = np.cumsum(self.durations) #サンプル長の累積和

    def init_dataset(self, n_tokens, sample_length, cache_dir):
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{self.audio_files_dir}', ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        logging.info(f"Found {len(files)} files. Getting durations")
        cache = cache_dir
        durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in files])  # Could be approximate　duration in sample_length
        self.filter(files, durations)

        if self.labels:
            self.labeller = Labeller(n_tokens, sample_length)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index] # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length: # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start: # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert start <= offset <= end - self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        return index, offset
    
    def get_song_chunk(self, index, offset, test=False):
        filepath, total_length = self.files[index], self.durations[index]
        song_id = os.path.splitext(os.path.basename(filepath))[0]
        data, sr = load_audio(filepath, sr=self.sr, offset=offset, duration=self.sample_length)
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        
        if not self.train_semantic:
            codec = AudioTokenizer(self.device)
            if data.size(-1) / sr > 15:
                raise ValueError(f"Prompt too long, expect length below 15 seconds, got {data / sr} seconds.")
            if data.size(0) == 2:
                data = data.mean(0, keepdim=True)
            _, codes = tokenize_audio(codec, (data, sr))
            audio_tokens = codes.transpose(2,1).cpu().numpy()  
            stok_music_path = f'{self.mert_unit_dir}/{song_id}_music.pt'
            music_unit = torch.load(stok_music_path)
            stok_lyric_path = f'{self.mert_unit_dir}/{song_id}_lyric.pt'
            lyric_unit = torch.load(stok_lyric_path)
            mert_feat_path = f'{self.mert_dir}/{song_id}.pt'
            mert_feat = torch.load(mert_feat_path)
        else:
            mert_unit_path = f'{self.mert_unit_dir}/{song_id}_acapella.pt'
            mert_unit = torch.load(mert_unit_path)
            audio_tokens = mert_unit.unsqueeze()
        
        if self.labels:
            with open(f'{self.lyrics_files_dir}/{song_id}.txt', 'r') as f:
                lyrics = f.read()
            genres = self.id_to_genres[song_id]
            info = self.id_to_info[song_id]
            lang = self.id_to_lang[song_id]
            metadata = self.id_to_metadata[song_id]
            tags = self.id_to_tags[song_id]
            
            labels = self.labeller.get_label(lyrics, genres, info, lang, metadata, tags, total_length, offset)
            labels['audio_features'] = audio_tokens
            labels['audio_features_lens'] = audio_tokens.shape[1]
            labels['audio'] = data
            if not self.train_semantic:
                labels['stok_music'] = music_unit
                labels['stok_lyric'] = lyric_unit
                labels['mert_feat'] = mert_feat
            return labels
        else:
            return data.T
    
    def get_dur(self, idx):
        return self.durations[idx]

    def get_item(self, item, test=False):
        index, offset = self.get_index_offset(item)
        return self.get_song_chunk(index, offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)
