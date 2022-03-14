#!/usr/bin/env python3
import pathlib
import sys
from multiprocessing.pool import ThreadPool, Pool
from datetime import timedelta, datetime
from dataclasses import dataclass
from threading import local
from typing import Iterable, Mapping, Union

import pandas as pd
import numpy as np
from rich import print
import librosa
import git
import warnings
import re

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
EXPECTED_SR = 128000
EXPECTED_OUTPUT_NUM_SAMPLES = 76800000
EXPECTED_NUM_CHANNLES = 1

class GLIDER(ICustomDataset):
    def __init__(self, pad = True):
        self._labels = GLIDER._todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
        self._audiofiles = GLIDER._todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))
        
        local_audiofiles = config.list_local_audiofiles()
        if len(local_audiofiles) != len(self._audiofiles):
            if len(local_audiofiles) > len(self._audiofiles):
                warnings.warn(f"There are local files that do not exist among the described in the csv at {config.PARSED_LABELS_PATH}")
            elif len(local_audiofiles) < len(self._audiofiles):
                warnings.warn(f"There are files described in the csv at {config.PARSED_LABELS_PATH} that do not exist in any subdirectory of {config.DATASET_DIRECTORY}")
        self._audiofiles.filename = [config.DATASET_DIRECTORY.joinpath(path) for path in self._audiofiles.filename.values]    
        self._audiofiles = self._audiofiles[self._audiofiles.filename.isin(local_audiofiles)]

        self._label_columns = self._labels.source_class.unique()
        self._classdict = self.classes()
        self._label_audiofiles()
        if config.VIRTUAL_DATASET_LOADING:
            warnings.warn("The environment variable VIRTUAL_DATASET_LOADING is set to True, meaning that the GLIDER dataset loading class will only simulate loading datasets from disk. Ensure that this variable is not set during training or inference, as it is only intended to be used during local development.")
            import socket
            if "idun-login" in socket.gethostname():
                raise Exception(f"GLIDER detected that the current hostname ({socket.gethostname()}) seems to correspond to the NTNU Idun computing cluster, while the VIRTUAL_DATASET_LOADING environment variable was set to True. This variable is only intended for local development and can cause unexpected results. This exception is raised to ensure that logs and model parameters are not overwritten using invalid/simulated data.")
        self._audiofiles.sort_values(by=["start_time"], inplace=True, ascending=True)
        self._pad = pad

    def _label_audiofiles(self):
        # Initialize label columns with missing values
        for col in self._label_columns:
            self._audiofiles[col] = None
            
        for idx in self._audiofiles.index:
            row = self._audiofiles.loc[idx]
            file_labels = self._labels[(self._labels.start_time <= row.end_time) & (self._labels.end_time >= row.start_time)]
            for label in file_labels.source_class.values:
                self._audiofiles.loc[idx, label] = config.POSITIVE_INSTANCE_CLASS_LABEL
        
        for col in self._label_columns:
            self._audiofiles[col] = self._audiofiles[col].fillna(config.NEGATIVE_INSTANCE_CLASS_LABEL)

    def _getitem_async(self, index):
        file = self._audiofiles.iloc[index]
        with ThreadPool(processes=2) as pool:
            async_labels_result = pool.apply_async(self._get_labels, (file, ))
            samples, sr = np.zeros(file.num_samples), file.sampling_rate
                
            if not config.VIRTUAL_DATASET_LOADING:
                async_samples_result = pool.apply_async(librosa.load, (file.filename, ), {"sr": None})
                samples, sr = async_samples_result.get()

            if self._pad:
                new_end_dt, padded_samples = self._ensure_equal_length(samples, sr, file.end_time, to_length=int(self._audiofiles.num_samples.max()))
                file.end_time = new_end_dt
                samples = padded_samples

            labels = async_labels_result.get()
            return LabeledAudioData(
                index,
                pathlib.Path(file.filename), 
                file.num_channels, 
                sr, 
                file.start_time, 
                file.end_time,
                samples, 
                labels,
                self._classdict)

    def _ensure_equal_length(self, samples: Iterable[float], sr: int, end_time: datetime, to_length: int) -> tuple[datetime, Iterable[float]]:
        if len(samples) < to_length:
            zeros = np.zeros(to_length - len(samples))
            padded_samples = np.concatenate((samples, zeros), axis=0)
            extra_duration_seconds = (to_length - len(samples)) / sr
            
            end_time = end_time + timedelta(seconds = extra_duration_seconds)
            samples = padded_samples
            
        elif len(samples) > to_length:
            duration_difference_seconds = (len(samples) - to_length) / sr
            end_time = end_time - timedelta(seconds = duration_difference_seconds)
            samples = samples[:to_length]
        
        return end_time, samples

    def __getitem__(self, index: int) -> LabeledAudioData:
        return self._getitem_async(index)
            
    def _get_labels(self, file):
        return self._labels[(self._labels.start_time <= file.end_time) & (self._labels.end_time >= file.start_time)]
            
    def __len__(self) -> int:
        return len(self._audiofiles)

    def classes(self) -> Mapping[str, int]:
        """BasicDataset.classes 'abstract' method implementation. Returns a dictionary of classname: class_index value.

        Returns:
            dict: dictionary of classname and corresponding class index in label matrix. Class index corresponds to the index along the first (zero-th) axis of the LabeledAudioExample.fill_labels() output matrix. 
        """
        return {self._label_columns[idx]: idx for idx in range(len(self._label_columns))}

    def example_shapes(self) -> Iterable[tuple[int, int]]:
        if self._pad:
            output_shape = (self._audiofiles.num_channels.max(), self._audiofiles.num_samples.max())
            return [output_shape for i in range(len(self._audiofiles))]
        else:
            shapes_as_lists = self._audiofiles[["num_channels", "num_samples"]].to_numpy()
            shapes_as_tuples = [tuple(item) for item in shapes_as_lists]
            return shapes_as_tuples

    def _todatetime(df):
        for col in df.columns:
            if "time" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def __str__(self):
        return f"GLIDER(BasicDataset) object with {len(self)} instances"

if __name__ == "__main__":
    dataset = GLIDER()

    for i in range(len(dataset)):
        data = dataset[i]
        # print(data)

    