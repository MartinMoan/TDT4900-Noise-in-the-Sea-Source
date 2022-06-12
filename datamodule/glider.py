#!/usr/bin/env python3
import argparse
from copy import copy, deepcopy
from email.mime import audio
import pickle
from dataclasses import dataclass
import datetime
import multiprocessing
import sys
import pathlib
from tabnanny import filename_only
from typing import Mapping, Iterable, Optional, Tuple, Union, Dict, List, Any
from multiprocessing.pool import ThreadPool
import hashlib

import git
from importlib_metadata import distribution
import torch
import re
import torch.utils.data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from rich import print
import torchaudio
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision.transforms import Normalize
import librosa

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from datasets.glider.wavfileinspector import WaveFileInspector
from interfaces import ITensorAudioDataset, IAugment
from datasets.initdata import create_tensorset
from datasets.glider.audiodata import AudioData
from datasets.augments.augment import CombinedAugment, SpecAugment
import matplotlib.pyplot as plt
from datamodule import utils

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        class_values: Tuple[str],
        nmels: int,
        nfft: int,
        hop_length: int,
        positive_label_value: Any = 1.0,
        negative_label_value: Any = 0.0) -> None:
        super().__init__()
        required_columns = ["filepath", "offset", "duration_seconds"]
        required_columns.extend(list(class_values))
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
        self.data = data
        self.class_values = class_values
        self.nmels = nmels
        self.nfft = nfft
        self.hop_length = hop_length
        self.positive_label_value = positive_label_value
        self.negative_label_value = negative_label_value
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = self.data.iloc[index]
        print(index, self.data.index.values[index], len(self.data), np.max(self.data.index.values), audio)
        samples, sr = librosa.load(audio.filepath, sr=None, offset=audio.offset, duration=audio.duration_seconds)
        spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=self.nmels, n_fft=self.nfft, hop_length=self.hop_length))
        spectrogram = np.flip(spectrogram, axis=0)
        spectrogram = torch.tensor(np.array(spectrogram), dtype=torch.float32, requires_grad=False)
        spectrogram = spectrogram.view(1, *spectrogram.shape)

        idx = self.data.index.values[index]
        # labels = self.data.iloc[index, [*self.class_values]]
        # label = torch.tensor(labels.to_numpy(), dtype=torch.float32, requires_grad=False)
        label = torch.tensor(
            [1.0 if audio[classname] == self.positive_label_value else 0.0 for classname in self.class_values], 
            dtype=torch.float32, 
            requires_grad=False
        )
        return spectrogram, label

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, augment: Optional[IAugment] = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataset) * self.augment.branching() + len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = index // (self.augment.branching() + 1)
        ai = int(index % (self.augment.branching()))
        if index % (self.augment.branching() + 1) == 0:
            return self.dataset[real_idx]
        else:
            data, label = self.dataset[real_idx]
            augmented = self.augment(data, label)
            return augmented[ai]

class GLIDERDatamodule(pl.LightningDataModule):
    def __init__(
        self, 
        recordings: pd.DataFrame,
        labels: pd.DataFrame,
        nfft: int,
        nmels: int,
        hop_length: int,
        clip_duration: Optional[Union[float, int]] = None,
        clip_overlap: Optional[Union[float, int]] = 0.0,
        train_percentage: Optional[float] = 0.8,
        val_percentage: Optional[float] = 0.1,
        specaugment: Optional[bool] = True,
        seed: Optional[Union[float, int]] = 42,
        verbose: bool = False,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

        self.recordings = utils.ensure_recording_columns(recordings)
        duplicated = self.recordings.duplicated(subset=["filepath"])
        print(f"Found {len(self.recordings[duplicated])} duplicate files, dropping duplicates")
        self.recordings = self.recordings.drop_duplicates(subset=["filepath"], keep="first", inplace=False, ignore_index=True)
        
        nonexisting_recordings = self.recordings.filepath.apply(lambda filepath: not pathlib.Path(filepath).exists())
        if len(self.recordings[nonexisting_recordings]) > 0:
            print(f"There are {len(self.recordings[nonexisting_recordings])} out of {len(self.recordings)} recordings that does not exists on the local filesystem. Dropping these recordings.")
            self.recordings = self.recordings[~nonexisting_recordings]

        self.labels, self._class_values = utils.ensure_dataframe_columns(labels)
        self._class_values = tuple(np.sort(self._class_values, axis=0))
        if len(self.recordings.sample_rate.unique()) != 1:
            raise Exception(f"There are differing sampling rates used among the recordings. Found sampling rates: {self.recordings.sample_rate.unique()}")
        self.sample_rate = self.recordings.sample_rate.max()
        
        self.clip_duration = clip_duration
        self.clip_overlap = clip_overlap
        self.train_part = train_percentage
        self.val_part = val_percentage
        self.nfft = nfft
        self.nmels = nmels
        self.hop_length = hop_length
        self.specaugment = specaugment

        self._positive_label_value = True
        self._negative_label_value = False


        self.verbose = verbose
        if self.clipping_enabled:
            self.clips = self.get_clips()
            self.labeled_clips = self.apply_labels(self.clips, positive_value=self._positive_label_value, negative_value=self._negative_label_value)
        else:
            self.labeled_recordings = self.apply_labels(self.recordings, positive_value=self._positive_label_value, negative_value=self._negative_label_value)
        
        self.label_distributions = self.group_by_labels(self.audio)
        self.seed = 20220621

    def setup(self, stage: Optional[str] = None):
        subsets = [distribution["subset"] for distribution in self.label_distributions]
        d = [len(distribution["subset"]) for distribution in self.label_distributions]
        n_in_smallest_subset = np.min(d)
        # Training set is balanced, with equal number of instances ber combination of labels
        n_for_training_per_class = int(n_in_smallest_subset * self.train_part)
        # Validation and test sets should have the same distribution, and not be balanced
        rng = np.random.default_rng(seed=self.seed)
        train_indeces = np.array([])
        for subset in subsets:
            indeces = subset.index.values
            for_training = rng.choice(indeces, size=n_for_training_per_class, replace=False)
            train_indeces = np.concatenate((train_indeces, for_training))

        train = self.audio.loc[train_indeces]
        duplicate = train.duplicated(subset=["filepath", "start_time", "end_time", "offset"])
        if len(train[duplicate]) > 0:
            raise Exception(f"There are {len(train[duplicate])} duplicate clips in the training set")

        remaining_distributions = [subset.loc[~subset.index.isin(train_indeces)] for subset in subsets]
        # Draw self.val_part from each subset, and remaining goes to testing
        validation_indeces = np.array([])
        for subset in remaining_distributions:
            indeces = subset.index.values
            for_validation = rng.choice(indeces, size=int(len(subset) * self.val_part), replace=False)
            validation_indeces = np.concatenate((validation_indeces, for_validation))
        
        val = self.audio.loc[validation_indeces]
        test = self.audio.loc[~self.audio.index.isin(validation_indeces) & ~self.audio.index.isin(train_indeces)]

        total_in_splits = np.sum([len(train), len(val), len(test)])
        if total_in_splits != len(self.audio):
            raise Exception(f"Unexpected total number of examples in train, val and test splits. Expected {len(train)} train + {len(val)} val + {len(test)} test = {len(self.audio)}, but found {total_in_splits}")

        duplicate = train.duplicated(subset=["filepath", "start_time", "end_time", "offset"])
        if len(train[duplicate]) > 0:
            raise Exception(f"There are {len(train[duplicate])} duplicated examples in train set")
        
        duplicate = val.duplicated(subset=["filepath", "start_time", "end_time", "offset"])
        if len(val[duplicate]) > 0:
            raise Exception(f"There are {len(val[duplicate])} duplicated examples in val set")
        
        duplicate = test.duplicated(subset=["filepath", "start_time", "end_time", "offset"])
        if len(test[duplicate]) > 0:
            raise Exception(f"There are {len(test[duplicate])} duplicated examples in test set")

        combined = pd.concat((train, test, val), ignore_index=True)
        duplicate = combined.duplicated(subset=["filepath", "start_time", "end_time", "offset"])
        if len(combined[duplicate]) > 0:
            raise Exception(f"There are {len(combined[duplicate])} duplicated examples in train, val and test sets (combined)")

        self.train_df = train
        self.val_df = val
        self.test_df = test

        self.train = AudioDataset(
            self.train_df, 
            self._class_values, 
            nmels=128, 
            nfft=3200, 
            hop_length=1280, 
            positive_label_value=self._positive_label_value, 
            negative_label_value=self._negative_label_value
        )
        indeces = rng.integers(0, len(self.train), 10)
        for i in indeces:
            self.train[i]
            print()

    @property
    def audio(self) -> pd.DataFrame:
        # return self.clips if self.clipping_enabled else self.recordings
        return self.labeled_clips if self.clipping_enabled else self.labeled_recordings

    @property
    def clipping_enabled(self) -> bool:
        return self.clip_duration is not None

    def apply_labels(self, audio_df: pd.DataFrame, positive_value: Any = True, negative_value: Any = False) -> pd.DataFrame:
        for cls in self._class_values:
            audio_df[cls] = negative_value
        for i in range(len(self.labels)):
            label = self.labels.iloc[i]
            audio = audio_df[(audio_df.start_time <= label.end_time) & (audio_df.end_time >= label.start_time)]
            if len(audio) > 0:
                audio_df.loc[audio.index, label.source_class] = positive_value
        return audio_df

    def get_clips(self):
        # df = pd.read_csv("clips.csv")
        # df.start_time = pd.to_datetime(df.start_time, errors="coerce")
        # df.end_time = pd.to_datetime(df.end_time, errors="coerce")
        # return df
        samples_per_clip = int(self.clip_duration * self.sample_rate)
        overlapping_samples = int(self.clip_overlap * self.sample_rate)
        clip_data = []
        for i in tqdm(range(len(self.recordings))):
            recording = self.recordings.iloc[i]
            num_clips_in_file = int((recording.num_samples - overlapping_samples) // (samples_per_clip - overlapping_samples))
            for j in range(num_clips_in_file):
                offset = (samples_per_clip - overlapping_samples) * j / self.sample_rate # offset in seconds
                clip = {col: recording[col] for col in self.recordings.columns}
                clip["file_start_time"] = clip["start_time"]
                clip["file_end_time"] = clip["end_time"]
                clip["file_duration_seconds"] = clip["duration_seconds"]
                clip["start_time"] = recording.start_time + datetime.timedelta(seconds=offset)
                clip["end_time"] = clip["start_time"] + datetime.timedelta(seconds=self.clip_duration)
                clip["duration_seconds"] = (clip["end_time"] - clip["start_time"]).seconds
                clip["offset"] = offset
                clip_data.append(clip)
        return pd.DataFrame(data=clip_data)

    def __len__(self) -> int:
        return len(self.audio)

    def _read_params(self, index: int) -> Tuple[pd.DataFrame, float]:
        if self.clipping_enabled:
            file_index, offset_seconds = self.clips[index]
            recording = self.recordings[file_index]
            return recording, offset_seconds
        else:
            file_index, offset_seconds = index, 0
            recording = self.recordings[index]
            return recording, offset_seconds

    def get_samples(self, index: int) -> Tuple[np.ndarray, int]:
        recording, offset_seconds = self._read_params(index)
        samples, sr = librosa.load(recording.filepath, sr=None, offset=offset_seconds, duration=self.clip_duration)
        return samples, sr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        recording, offset_seconds = self._read_params(index)
        samples, sr = librosa.load(recording.filepath, sr=None, offset=offset_seconds, duration=self.clip_duration)
        raise NotImplementedError

    def _label_combinations(self, positive_value: Optional[Any] = True, negative_value: Optional[any] = False):
        C = len(self._class_values)
        N = 2**C
        helper = np.full((N, C), fill_value=negative_value)
        frequency = 1
        for c in range(len(self._class_values)):
            current = positive_value
            for i in range(N):
                if i % frequency == 0:
                    current = negative_value if current == positive_value else positive_value
                helper[i, c] = current
            frequency *= 2
        output = []
        for i in range(N):
            data = {self._class_values[c]:  helper[i, c] for c in range(len(self._class_values))}
            output.append(data)
        return output

    def group_by_labels(self, labeled_audio: pd.DataFrame):
        combinations = self._label_combinations(positive_value=self._positive_label_value, negative_value=self._negative_label_value)
        for combination in combinations:
            subset = labeled_audio[np.logical_and.reduce([(labeled_audio[k] == v) for k,v in combination.items()])]
            combination["subset"] = subset
        return combinations

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    find = subparsers.add_parser("find", help="Find audiodfiles recursively, get their metadata such as start_time, end_time and sampling rate, and store to csv")
    find.add_argument("root_path", type=pathlib.Path, help="Root path to search for .wav files from")
    find.add_argument("-o", "--output", type=pathlib.Path, help="Output path for audiofile csv", default=pathlib.Path(__file__).parent.joinpath("metadata.csv"))
    find.add_argument("-p", "--processes", type=int, default=multiprocessing.cpu_count(), help="Number of processes to use when reading file metadata")
    find.add_argument("-l", "--limit", type=int, help="Limit number of files to parse, usefull for debugging")
    
    load = subparsers.add_parser("load", help="Load AudioDataset using provided audiofile and label CSVs")
    load.add_argument("-l", "--labels", type=pathlib.Path, help="Path to labels csv")
    load.add_argument("-a", "--audiofiles", type=pathlib.Path, help="Path to audiofile metadata csv")

    args = parser.parse_args()
    if args.command is None:
        raise ValueError("Missing command argument")
    return args

def preload(args):
    path = args.root_path.resolve()
    if not path.exists():
        raise ValueError(f"Path {path} does not exists")
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory.")

    output_path = args.output.resolve()
    if output_path.suffix != ".csv":
        raise ValueError(f"Output path {output_path} is not .csv file")

    filepaths = utils.get_audiofiles([path], verbose=args.verbose)
    print(f"Found {len(filepaths)} .wav files in local path: {path}")
    recordings = utils.get_audiofile_information(audiofiles=filepaths, processes=args.processes, limit=args.limit)
    print(recordings)
    print(f"Saving audiofiles metadata to: {output_path}")
    recordings.to_csv(output_path, index=False)

if __name__ == "__main__":
    args = init()
    if args.command == "find":
        preload(args)
    elif args.command == "load":
        labels = pd.read_csv(args.labels)
        recordings = pd.read_csv(args.audiofiles)
        data = GLIDERDatamodule(
            recordings=recordings,
            labels=labels,
            verbose=True,
            clip_duration=10.0,
            clip_overlap=4.0
        )
        data.setup()

