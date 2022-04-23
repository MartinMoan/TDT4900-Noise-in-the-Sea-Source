#!/usr/bin/env python3
import pathlib
import sys
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
from datetime import timedelta, datetime
from dataclasses import dataclass
from threading import local
from typing import Iterable, Mapping, Union
import math

import pandas as pd
import numpy as np
from rich import print
import librosa
from scipy.io import wavfile
import git
import warnings
import re

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces.ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
from interfaces import ILogger
from tracking.logger import Logger
EXPECTED_SR = 128000
EXPECTED_OUTPUT_NUM_SAMPLES = 76800000
EXPECTED_NUM_CHANNLES = 1

class GLIDER(ICustomDataset):
    def __init__(self, clip_duration_seconds: float = 10.0, overlap_seconds=0.0, verbose = False, suppress_warnings = None, logger: ILogger = Logger()):
        self.logger = logger
        self._labels = GLIDER._todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
        self._audiofiles = GLIDER._todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))
        expected_num_samples = self._audiofiles.num_samples.max()
        self._audiofiles = self._audiofiles[(self._audiofiles.num_samples == expected_num_samples)]
        
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
        self._pad = True
        self._clip_duration_seconds = clip_duration_seconds
        self._file_duration_seconds = self._audiofiles.num_samples.max() / self._audiofiles.sampling_rate.max()
        self._num_clips = int((len(self._audiofiles) * self._file_duration_seconds) / self._clip_duration_seconds)
        self._verbose = verbose
        self._suppress_warnings = config.ENV == "env" if suppress_warnings is None else suppress_warnings

        self.clips_per_file = self._num_clips / len(self._audiofiles)


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

    def __getitem__(self, index: int) -> LabeledAudioData:
        clip_index = index
        file_index = int(clip_index / self.clips_per_file)
        file = self._audiofiles.iloc[file_index]

        first_clip_in_file_clip_index = file_index * self.clips_per_file
        nth_clip_in_file = clip_index - first_clip_in_file_clip_index
        clip_start_offset_seconds = nth_clip_in_file * self._clip_duration_seconds

        if clip_start_offset_seconds >= (file.end_time - file.start_time).seconds:
            return self.__getitem__(index + 1) # This one does not spark joy...

        audio_data = LabeledAudioData(
            clip_index,
            file.filename,
            file.num_channels,
            file.sampling_rate,
            file.start_time,
            file.end_time,
            self._clip_duration_seconds,
            clip_start_offset_seconds,
            all_labels = self._labels,
            labels_dict = self.classes()
        )
        
        if self._verbose:
            self.logger.log(f"{index} / {len(self)}", audio_data)
            
        return audio_data
            
    def _get_labels(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        return self._labels[(self._labels.start_time <= end_time) & (self._labels.end_time >= start_time)]
            
    def __len__(self) -> int:
        return self._num_clips

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

    def __repr__(self):
        raise NotImplementedError

if __name__ == "__main__":
    dataset = GLIDER(clip_duration_seconds=10.0, verbose=True, suppress_warnings=True)
    
    for i in range(len(dataset)):
        data = dataset[i]

    