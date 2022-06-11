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

def typecheck_audiofile_directory(audiofile_directory):
    audiofile_directories = []
    if isinstance(audiofile_directory, (np.ndarray, list, tuple)):
        for path in audiofile_directory:
            if isinstance(path, pathlib.Path):
                audiofile_directories.append(path)
            elif isinstance(path, str):
                path = pathlib.Path(path)
                audiofile_directories.append(pathlib.Path(path))
            else:
                raise TypeError(f"audiofile path value has invalid type, expected string or pathlib.Path but received {type(path)}")
    elif isinstance(audiofile_directory, pathlib.Path):
        audiofile_directories = [audiofile_directory]
    elif isinstance(audiofile_directory, str):
        audiofile_directories = [pathlib.Path(audiofile_directory)]
    else:
        raise TypeError(f"audiofile_directory has incorrect type {type(audiofile_directory)}")
    return audiofile_directories

def ensure_dirs_exists(directories: Iterable[Union[pathlib.Path, str]]) -> None:
    for dir in directories:
        path = pathlib.Path(dir)
        if not path.exists():
            raise ValueError(f"Path {path} does not exists")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
    return directories

def ensure_dataframe_columns(df: pd.DataFrame) -> None:    
    required_columns = ["source_class", "start_time", "end_time"]
    for column in required_columns:
        if column not in df.columns:
            raise Exception(f"labels DataFrame is missing required column {column}")
    df.start_time = pd.to_datetime(df.start_time, errors="coerce")
    df.end_time = pd.to_datetime(df.end_time, errors="coerce")
    class_values = df.source_class.unique()
    return df, class_values

def list_audiofiles(path: pathlib.Path):
    return list(path.glob("**/*.wav"))

def get_audiofiles(directories: Iterable[pathlib.Path], verbose: bool = False) -> Iterable[pathlib.Path]:
    output = []
    if verbose:
        print("Finding audiofiles...")
        for dir in tqdm(directories):
            output += list_audiofiles(dir)
    else:
        for dir in directories:
            output += list_audiofiles(dir)
    return output

def parse_audio_filename(path: pathlib.Path) -> Dict[str, Union[datetime.datetime, pathlib.Path, Exception]]:
    filename_information = re.search(r"([^_]+)_([0-9]{3})_([0-9]{6}_[0-9]{6})\.wav", path.name)
    # dive_number = filename_information.group(1)
    # identification_number = filename_information.group(2)
    timestring = filename_information.group(3)
    start_time = None
    error = None
    try:
        start_time = datetime.datetime.strptime(timestring, "%y%m%d_%H%M%S")
    except Exception as ex:
        error = ex
        start_time = datetime.datetime(year=1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return dict(
        filepath=path,
        start_time=start_time,
        error=error
    )

def parse_filenames(audiofiles: Iterable[pathlib.Path]) -> Iterable[Dict[str, Union[datetime.datetime, pathlib.Path, Exception]]]:
    return [parse_audio_filename(path) for path in audiofiles]

def get_info(filename_information):
    if filename_information["error"] is not None:
        return None

    samples, sr = librosa.load(filename_information["filepath"], sr=None)
    duration_seconds = len(samples) / sr
    return dict(
        filepath=filename_information["filepath"],
        start_time=filename_information["start_time"],
        end_time=filename_information["start_time"] + datetime.timedelta(seconds=duration_seconds),
        sample_rate=sr,
        num_samples=len(samples),
        duration_seconds=duration_seconds
    )
    
def chunk(iterable, func, position):
    output = []
    with tqdm(total=len(iterable), position=position, leave=False) as pbar:
        for i in range(len(iterable)):
            pbar.update(1)
            try:
                data = func(iterable[i])
                if data is not None:
                    output.append(data)
            except Exception as ex:
                print(f"An exception occured when reading the file {iterable[i]}, skipping file: {ex}")
                continue
        return output

def get_audiofile_information(audiofiles: Iterable[pathlib.Path], processes: int = multiprocessing.cpu_count(), limit: Optional[int] = None) -> pd.DataFrame:
    filename_info = parse_filenames(audiofiles)
    max = limit if limit is not None else len(filename_info)
    chunksize = max / processes
    with ThreadPool(processes=processes) as pool:
        tasks = []
        print("Initializing jobs...")
        for i in range(processes + 1):
            start = int(i*chunksize)
            stop = min(max, int((i+1)*chunksize))
            task = pool.apply_async(chunk, args=(filename_info[start:stop], get_info, i))
            tasks.append(task)
        
        print(f"Performing {len(tasks)} jobs...")
        result = np.array([])
        for task in tasks:
            result = np.concatenate((result, task.get()))
        return pd.DataFrame(data=list(result))

def get_metadata(audiofiles: Iterable[pathlib.Path], cache_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = pathlib.Path(__file__).parent.joinpath("metadata")
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=False)

    sorted_audiofiles = np.sort(audiofiles, axis=0)
    hasher = hashlib.sha256()
    hasher.update(repr(sorted_audiofiles).encode())
    hash = hasher.hexdigest()
    pickle_filename = f"{str(hash)}.pickle"
    pickle_path = cache_dir.joinpath(pickle_filename).absolute()
    cached_metadata_filepaths = [path for path in cache_dir.glob("**/*.pickle")]

    if pickle_path in cached_metadata_filepaths:
        print(f"Loading from cache {pickle_path}")
        with open(pickle_path, "rb") as binary_file:
            return pickle.load(binary_file)
    else:
        output = get_audiofile_information(audiofiles=audiofiles)
        with open(pickle_path, "wb") as binary_file:
            pickle.dump(output, binary_file)
            return output

def ensure_recording_columns(recordings_df):
    required_columns = ["filepath", "start_time", "end_time"]
    for column in required_columns:
        if column not in recordings_df.columns:
            raise Exception(f"labels DataFrame is missing required column {column}")
    recordings_df.start_time = pd.to_datetime(recordings_df.start_time, errors="coerce")
    recordings_df.end_time = pd.to_datetime(recordings_df.end_time, errors="coerce")
    return recordings_df

class GLIDERDatamodule(pl.LightningDataModule):
    def __init__(
        self, 
        recordings: pd.DataFrame,
        labels: pd.DataFrame,
        clip_duration: Optional[Union[float, int]] = None,
        clip_overlap: Optional[Union[float, int]] = 0.0,
        verbose: bool = False,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

        self.recordings = ensure_recording_columns(recordings)
        duplicated = self.recordings.duplicated(subset=["filepath"])
        print(f"Found {len(self.recordings[duplicated])} duplicate files, dropping duplicates")
        self.recordings = self.recordings.drop_duplicates(subset=["filepath"], keep="first", inplace=False, ignore_index=True)
        print(self.recordings["duration_seconds"].sum() / (clip_duration - clip_overlap))

        self.labels, self.class_values = ensure_dataframe_columns(labels)
        if len(self.recordings.sample_rate.unique()) != 1:
            raise Exception(f"There are differing sampling rates used among the recordings. Found sampling rates: {self.recordings.sample_rate.unique()}")
        self.sample_rate = self.recordings.sample_rate.max()
        self.clip_duration = clip_duration
        self.clip_overlap = clip_overlap
        self._positive_label_value = True
        self._negative_label_value = False
        if self.clipping_enabled:
            self.clips = self.get_clips()
            self.labeled_clips = self.apply_labels(self.clips)
        else:
            self.labeled_recordings = self.apply_labels(self.recordings, positive_value=self._positive_label_value, negative_value=self._negative_label_value)
        
        self.label_distributions = self.group_by_labels(self.audio)
        self.train_part, self.val_part, self.test_part = 0.64, 0.16, 0.2

    def setup(self, stage: Optional[str] = None):
        subsets = [distribution["subset"] for distribution in self.label_distributions]
        d = [len(distribution["subset"]) for distribution in self.label_distributions]
        n_in_smallest_subset = np.min(d)
        print(n_in_smallest_subset)
        print(d)
        print(len(self.audio))
        print(np.sum(d))
        n_for_training = int(n_in_smallest_subset * len(self.class_values) * self.train_part)
        n_for_training_per_class = int(n_in_smallest_subset * self.train_part)
        training_clips = pd.concat([subset.iloc[:n_for_training_per_class] for subset in subsets])
        print(training_clips)
        

    @property
    def audio(self) -> pd.DataFrame:
        # return self.clips if self.clipping_enabled else self.recordings
        return self.labeled_clips if self.clipping_enabled else self.labeled_recordings

    @property
    def clipping_enabled(self) -> bool:
        return self.clip_duration is not None

    def apply_labels(self, audio_df: pd.DataFrame, positive_value: Any = True, negative_value: Any = False) -> pd.DataFrame:
        for cls in self.class_values:
            audio_df[cls] = negative_value
        for i in range(len(self.labels)):
            label = self.labels.iloc[i]
            audio = audio_df[(audio_df.start_time <= label.end_time) & (audio_df.end_time >= label.start_time)]
            if len(audio) > 0:
                audio_df.loc[audio.index, label.source_class] = positive_value
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
        if self.clipping_enabled:
            return len(self.clips)
        return len(self.recordings)

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
        C = len(self.class_values)
        N = 2**C
        helper = np.full((N, C), fill_value=negative_value)
        frequency = 1
        for c in range(len(self.class_values)):
            current = positive_value
            for i in range(N):
                if i % frequency == 0:
                    current = negative_value if current == positive_value else positive_value
                helper[i, c] = current
            frequency *= 2
        output = []
        for i in range(N):
            data = {self.class_values[c]:  helper[i, c] for c in range(len(self.class_values))}
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

    filepaths = get_audiofiles([path], verbose=args.verbose)
    print(f"Found {len(filepaths)} .wav files in local path: {path}")
    recordings = get_audiofile_information(audiofiles=filepaths, processes=args.processes, limit=args.limit)
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

