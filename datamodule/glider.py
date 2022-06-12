#!/usr/bin/env python3
import argparse
import datetime
import multiprocessing
import os
import pathlib
import sys
from typing import Any, Callable, Literal, Optional, Tuple, Union, Dict

import git
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
import wandb
from pytorch_lightning.loggers import WandbLogger
from rich import print
from torchvision.transforms import Normalize
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from datasets.augments.augment import SpecAugment
from interfaces import IAugment

from datamodule import utils


def no_op_transform(*args):
    return args

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        class_values: Tuple[str],
        nmels: int,
        nfft: int,
        hop_length: int,
        positive_label_value: Any = 1.0,
        negative_label_value: Any = 0.0, 
        transforms: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor]]] = None) -> None:
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
        self.transforms = transforms if transforms is not None else no_op_transform

    def audiodata(self, index: int):
        return self.data.iloc[index]
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = self.data.iloc[index]
        samples, sr = librosa.load(audio.filepath, sr=None, offset=audio.offset, duration=audio.duration_seconds)
        spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=self.nmels, n_fft=self.nfft, hop_length=self.hop_length))
        spectrogram = np.flip(spectrogram, axis=0)
        spectrogram = torch.tensor(np.array(spectrogram), dtype=torch.float32, requires_grad=False)
        spectrogram = spectrogram.view(1, *spectrogram.shape)

        # idx = self.data.index.values[index]
        # labels = self.data.iloc[index, [*self.class_values]]
        # label = torch.tensor(labels.to_numpy(), dtype=torch.float32, requires_grad=False)
        label = torch.tensor(
            [1.0 if audio[classname] == self.positive_label_value else 0.0 for classname in self.class_values], 
            dtype=torch.float32, 
            requires_grad=False
        )
        return self.transforms(spectrogram), label

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Union[torch.utils.data.Dataset, AudioDataset], augment: Optional[IAugment] = None) -> None:
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

    def audiodata(self, index: int):
        if not isinstance(self.dataset, AudioDataset):
            raise NotImplementedError
        
        real_idx = index // (self.augment.branching() + 1)
        return self.dataset.audiodata(real_idx)

class GLIDERDatamodule(pl.LightningDataModule):
    def __init__(
        self, 
        recordings: pd.DataFrame,
        labels: pd.DataFrame,
        nfft: int,
        nmels: int,
        hop_length: int,
        batch_size: int,
        clip_duration: Optional[Union[float, int]] = None,
        clip_overlap: Optional[Union[float, int]] = 0.0,
        train_percentage: Optional[float] = 0.8,
        val_percentage: Optional[float] = 0.1,
        duplicate_error: Optional[Literal['raise', 'drop']] = 'drop',
        specaugment: Optional[bool] = True,
        max_time_mask_seconds: Optional[Union[float, int]] = None,
        specaugment_branching: Optional[int] = 3,
        max_mel_masks: Optional[int] = None,
        num_workers: Optional[int] = os.cpu_count(),
        seed: Optional[Union[float, int]] = 42,
        normalize: Optional[bool] = True,
        mu: Optional[Union[float, int]] = None,
        sigma: Optional[Union[float, int]] = None,
        normalize_limit: Optional[float] = 1.0,
        verbose: Optional[bool] = False,
        wandblogger: Optional[WandbLogger] = None,
        track_n_examples: Optional[int] = 10,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        if max_mel_masks <= 0.0 or max_mel_masks > nmels:
            raise ValueError(f"max_mel_masks has invalid value, must be in range [1, nmels]")

        if duplicate_error not in ["raise", "drop"]:
            raise ValueError(f"Argument duplicate_error has invalid value, expected 'raise' or 'drop' but received {type(duplicate_error)} with value {repr(duplicate_error)}")

        if normalize_limit > 1.0 or normalize_limit < 0.0:
            raise ValueError(f"Argument 'normalize_limit' has invalid value, must be float in range 0-1 but received value {normalize_limit}")

        self.recordings = utils.ensure_recording_columns(recordings)
        duplicated = self.recordings.duplicated(subset=["filepath"])
        if len(self.recordings[duplicated]) > 0:
            if duplicate_error == 'drop':
                print(f"Found {len(self.recordings[duplicated])} duplicate files, dropping duplicates")
                self.recordings = self.recordings.drop_duplicates(subset=["filepath"], keep="first", inplace=False, ignore_index=True)
            elif duplicate_error == 'raise':
                raise Exception(f"There are {len(self.recordings[duplicated])} duplicate recordings in the provided recordings argument: {self.recordings[duplicated]}")
        
        nonexisting_recordings = self.recordings.filepath.apply(lambda filepath: not pathlib.Path(filepath).exists())
        if len(self.recordings[nonexisting_recordings]) > 0:
            print(f"There are {len(self.recordings[nonexisting_recordings])} out of {len(self.recordings)} recordings that does not exists on the local filesystem. Dropping these recordings.")
            self.recordings = self.recordings[~nonexisting_recordings]
        
        if len(self.recordings.sample_rate.unique()) != 1:
            raise Exception(f"There are differing sampling rates used among the recordings. Found sampling rates: {self.recordings.sample_rate.unique()}")

        self.sample_rate = self.recordings.sample_rate.max()
        self.labels, self._class_values = utils.ensure_dataframe_columns(labels)
        self._class_values = tuple(np.sort(self._class_values, axis=0))
        self.class_names = list(self._class_values) # This should be consumed by users, not self._class_values
        self.batch_size = batch_size
        self.clip_duration = clip_duration
        self.clip_overlap = clip_overlap
        self.train_part = train_percentage
        self.val_part = val_percentage
        self.nfft = nfft
        self.nmels = nmels
        self.hop_length = hop_length
        self.specaugment = specaugment
        self.num_workers = num_workers
        self.max_mel_masks = max_mel_masks if max_mel_masks is not None else nmels // 8
        self.max_time_mask_seconds = max_time_mask_seconds
        self.specaugment_branching = specaugment_branching
        self.seed = seed
        self._positive_label_value = True
        self._negative_label_value = False
        self.verbose = verbose
        self.normalize = normalize
        self.normalizer = None
        self.mu, self.sigma = mu, sigma
        self.normalize_limit = normalize_limit
        self.logger = wandblogger
        self.track_n_examples = track_n_examples
        self.setup_complete = False
        self._tracking_complete = False

        if self.clipping_enabled:
            self.clips = self.get_clips()
            self.labeled_clips = self.apply_labels(self.clips, positive_value=self._positive_label_value, negative_value=self._negative_label_value)
        else:
            self.labeled_recordings = self.apply_labels(self.recordings, positive_value=self._positive_label_value, negative_value=self._negative_label_value)
        
        self.label_distributions = self.group_by_labels(self.audio)

    def loggables(self) -> Dict[str, Any]:
        if not self.setup_complete:
            self.setup()
        distributions = {}
        for subset in self.label_distributions:
            keys = [key for key in subset.keys() if key != "subset"]
            label = ", ".join(keys)
            distributions[label] = subset["subset"]
        n_per_label = {label: len(subset) for label, subset in distributions.items()}

        to_log = {
            "loader_sizes": {
                "train_loader_size": len(self.train_dataloader()),
                "val_loader_size": len(self.val_dataloader()),
                "test_loader_size": len(self.test_dataloader())
            },
            "augmented_sizes": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test)
            },
            "original_sizes": {
                "train": len(self.train_df),
                "val": len(self.val_df),
                "test": len(self.test_df)
            },
            "dataset_parameters": {
                "label_distributions": n_per_label,
                "n_recordings": len(self.recordings),
                "n_audio": len(self.audio),
                "class_values": self._class_values,
                "batch_size": self.batch_size,
                "clip_duration": self.clip_duration,
                "clip_overlap": self.clip_overlap,
                "train_part": self.train_part,
                "val_part": self.val_part,
                "nfft": self.nfft,
                "nmels": self.nmels,
                "hop_length": self.hop_length,
                "specaugment": self.specaugment,
                "num_workers": self.num_workers,
                "max_mel_masks": self.max_mel_masks,
                "max_time_mask_seconds": self.max_time_mask_seconds,
                "specaugment_branching": self.specaugment_branching,
                "seed": self.seed,
                "positive_label_value": self._positive_label_value,
                "negative_label_value": self._negative_label_value,
                "verbose": self.verbose
            }
        }
        return to_log

    def _track(self) -> None:
        slurm_procid = int(os.environ.get("SLURM_PROCID", default=-1))
        print(f"Found slurm procid: {slurm_procid}")
        if not (slurm_procid == -1 or slurm_procid == 0):
            # If -1, not running in slurm managed environment, so we should log
            # If 0, running in slurm managed environment, and rank is 0, so we should log
            # In all other cases, we should not log
            return

        if self.logger is None:
            return
        if not self.setup_complete:
            self.setup() 
            return # self.setup() calls self._track(), so will already be done
        if self._tracking_complete:
            return
        
        self._track_subset("train", self.train, n_examples=self.track_n_examples)
        self._track_subset("val", self.val, n_examples=self.track_n_examples)
        self._track_subset("test", self.test, n_examples=self.track_n_examples)

        self.logger.experiment.config.update(self.loggables())
        self._tracking_complete = True

    def _track_subset(self, stage: str, dataset: AudioDataset, n_examples: int = 10) -> None:
        if n_examples <= 0 or len(dataset) < n_examples:
            return
        
        indeces = np.random.random_integers(0, len(dataset) - 1, n_examples)

        table = wandb.Table(columns=[
            "audio", 
            "spectrogram", 
            "source_class_specific", 
            "clip_classes",
            "dataset_classes", 
            "tensor_label", 
            "filename",
            "file_start_time",
            "file_end_time",
            "clip_offset_seconds",
            "clip_duration_seconds",
            "clip_index"
        ])

        for index in range(len(indeces)):
            i = indeces[index]
            X, Y = dataset[i]
            audiodata = dataset.audiodata(i)
            samples, sr = librosa.load(audiodata.filepath, sr=None, offset=audiodata.offset, duration=audiodata.duration_seconds)
            audio = wandb.Audio(samples, sample_rate=sr)
            spectrogram = wandb.Image(X, caption=f"{audiodata.filepath.name}")
            source_class_specific = audiodata.source_class_specific
            cls = audiodata[[*self._class_values]].to_dict()
            clip_classes = ", ".join([key for key, value in cls.items() if value == self._positive_label_value])
            datetimeformat = "%c"

            table.add_data(
                audio, 
                spectrogram, 
                source_class_specific, 
                clip_classes, 
                self._class_values, 
                str(Y.detach().numpy()),
                audiodata.filepath.name,
                audiodata.file_start_time.strftime(datetimeformat),
                audiodata.file_end_time.strftime(datetimeformat),
                audiodata.offset,
                audiodata.duration_seconds,
                i
            )
            if self.verbose:
                print(f"Logging dataset example {index} / {len(indeces)}")
        self.logger.experiment.log({f"{stage}_examples": table})

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

        random_state = np.random.RandomState(seed=self.seed)
        self.train_df = train.sample(frac=1, axis='index', replace=False, random_state=random_state) #Shuffles dataframe, important to avoid first batches to return same labels
        self.val_df = val.sample(frac=1, axis='index', replace=False, random_state=random_state)
        self.test_df = test.sample(frac=1, axis='index', replace=False, random_state=random_state)

        self.train = AudioDataset(
            data=self.train_df,
            class_values=self._class_values,
            nmels=self.nmels,
            nfft=self.nfft,
            hop_length=self.hop_length,
            positive_label_value=self._positive_label_value,
            negative_label_value=self._negative_label_value
        )
        if self.normalize:
            if self.mu is None or self.sigma is None:
                self.mu, self.sigma = utils.normalize(self.train, limit=int(len(self.train)*self.normalize_limit), randomize=self.seed)
            self.normalizer = Normalize(mean=self.mu, std=self.sigma, inplace=False)
            self.train.transforms = self.normalizer
            
        if self.specaugment:
            self.train_audiodataset = self.train
            self.train = AugmentedDataset(
                self.train_audiodataset,
                augment=SpecAugment(
                    branching=self.specaugment_branching,
                    nmels=self.nmels,
                    hop_length=self.hop_length,
                    max_time_mask_seconds=self.max_time_mask_seconds,
                    sr=self.sample_rate,
                    max_mel_masks=self.max_mel_masks,
                    max_fails=int(len(self.train_audiodataset) * 0.01)
                )
            )

        self.val = AudioDataset(
            data=self.val_df,
            class_values=self._class_values,
            nmels=self.nmels,
            nfft=self.nfft,
            hop_length=self.hop_length,
            positive_label_value=self._positive_label_value,
            negative_label_value=self._negative_label_value,
            transforms=self.normalizer
        )
        self.test = AudioDataset(
            data=self.test_df,
            class_values=self._class_values, 
            nmels=self.nmels,
            nfft=self.nfft,
            hop_length=self.hop_length,
            positive_label_value=self._positive_label_value,
            negative_label_value=self._negative_label_value,
            transforms=self.normalizer
        )
        self.setup_complete = True
        self._track()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            sampler=torch.utils.data.RandomSampler(self.train, replacement=False)
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            sampler=torch.utils.data.RandomSampler(self.val, replacement=False)
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            sampler=torch.utils.data.RandomSampler(self.test, replacement=False)
        )
    
    @property
    def audio(self) -> pd.DataFrame:
        return self.labeled_clips if self.clipping_enabled else self.labeled_recordings

    @property
    def clipping_enabled(self) -> bool:
        return self.clip_duration is not None

    def apply_labels(self, audio_df: pd.DataFrame, positive_value: Any = True, negative_value: Any = False) -> pd.DataFrame:
        div_cols = [col for col in self.labels.columns if col not in audio_df.columns and col != "source_class"]
        for col in div_cols:
            audio_df[col] = ""

        for cls in self._class_values:
            audio_df[cls] = negative_value
        for i in range(len(self.labels)):
            label = self.labels.iloc[i]
            audio = audio_df[(audio_df.start_time <= label.end_time) & (audio_df.end_time >= label.start_time)]
            
            if len(audio) > 0:
                audio_df.loc[audio.index, label.source_class] = positive_value
                for col in div_cols:
                    audio_df.loc[audio.index, col] = audio_df.loc[audio.index, col].astype(str) + f"{label[col]}, "
        return audio_df

    def get_clips(self):
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

    def group_by_labels(self, labeled_audio: pd.DataFrame):
        combinations = utils.label_combinations(class_values=self._class_values, positive_value=self._positive_label_value, negative_value=self._negative_label_value)
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

import matplotlib.pyplot as plt


def plot(id: int, spectrogram: torch.Tensor, labels: torch.Tensor):
    plt.imshow(spectrogram.squeeze().detach().numpy(), aspect="auto")
    plt.title(f"Id: {id} Labels: {labels.detach().numpy()}")
    plt.show()

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
            clip_duration=20.0,
            clip_overlap=0.0,
            nfft=3200,
            nmels=128,
            hop_length=1280,
            batch_size=8,
            train_percentage=0.8,
            val_percentage=0.1,
            duplicate_error="raise",
            specaugment=True,
            max_time_mask_seconds=2.0,
            specaugment_branching=3,
            max_mel_masks=128//8,
            num_workers=os.cpu_count(),
            seed=42,
            normalize=True,
            mu=-47.5545,
            sigma=13.5853
        )
        data.setup()
        # logger = WandbLogger(
        #     save_dir=str(config.HOME_PROJECT_DIR.absolute()),
        #     offline=False,
        #     project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
        #     entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
        #     config=vars(hyperparams), # These are added to wandb.init call as part of the config,
        #     tags=hyperparams.tracking_tags,
        #     notes=hyperparams.tracking_notes,
        #     name=hyperparams.tracking_name
        # )
        d = data.train_dataloader()
        indeces = np.random.randint(0, len(d), 10)
        for i, (X, Y) in enumerate(d):
            print(X.shape, Y, Y.shape)
        
        print(len(data.train_df))
        print(len(data.train))
        print(len(data.train_dataloader())*8)
        print(len(data.val_df))
        print(len(data.val))
        print(len(data.val_dataloader())*8)
        print(len(data.test_df))
        print(len(data.test))
        print(len(data.test_dataloader())*8)

