#!/usr/bin/env python3
import pathlib
import sys
from multiprocessing.pool import ThreadPool, Pool
from datetime import timedelta
from dataclasses import dataclass
from threading import local

import pandas as pd
import numpy as np
from rich import print
import librosa
import git
import warnings
import re

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from BasicDataset import BasicDataset
from audiodata import LabeledAudioData

class GLIDER(BasicDataset):
    def __init__(self):
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

    def _label_audiofiles(self):
        for idx in self._audiofiles.index:
            row = self._audiofiles.loc[idx]
            file_labels = self._labels[(self._labels.start_time <= row.end_time) & (self._labels.end_time >= row.start_time)]
            for label in file_labels.source_class.values:
                self._audiofiles.loc[idx, label] = config.POSITIVE_INSTANCE_CLASS_LABEL
        
        for col in self._label_columns:
            self._audiofiles[col] = self._audiofiles[col].fillna(config.NEGATIVE_INSTANCE_CLASS_LABEL)

    def _getitem_blocking(self, index):
        file = self._audiofiles.iloc[index]
        labels = self._get_labels(file)
        samples, sr = np.zeros(file.num_samples), file.sampling_rate
        if not config.VIRTUAL_DATASET_LOADING:
            samples, sr = librosa.load(file.filename, sr=None)
        return LabeledAudioData(
            index,
            pathlib.Path(file.filename), 
            file.num_channels, 
            sr, 
            file.num_samples, 
            file.start_time, 
            file.end_time,
            samples, 
            labels,
            self._classdict)

    def _getitem_async(self, index):
        file = self._audiofiles.iloc[index]
        with ThreadPool(processes=2) as pool:
            async_labels_result = pool.apply_async(self._get_labels, (file, ))
            samples, sr = np.zeros(file.num_samples), file.sampling_rate
            if not config.VIRTUAL_DATASET_LOADING:
                async_samples_result = pool.apply_async(librosa.load, (file.filename, ), {"sr": None})
                samples, sr = async_samples_result.get()

            labels = async_labels_result.get()
            return LabeledAudioData(
                index,
                pathlib.Path(file.filename), 
                file.num_channels, 
                sr,
                file.num_samples, 
                file.start_time, 
                file.end_time,
                samples, 
                labels,
                self._classdict)

    def __getitem__(self, index):
        return self._getitem_async(index)
            
    def _get_labels(self, file):        
        return self._labels[(self._labels.start_time <= file.end_time) & (self._labels.end_time >= file.start_time)]
            
    def __len__(self):
        return len(self._audiofiles)

    def classes(self) -> dict:
        """BasicDataset.classes 'abstract' method implementation. Returns a dictionary of classname: class_index value.

        Returns:
            dict: dictionary of classname and corresponding class index in label matrix. Class index corresponds to the index along the first (zero-th) axis of the LabeledAudioExample.fill_labels() output matrix. 
        """
        return {self._label_columns[idx]: idx for idx in range(len(self._label_columns))}

    def _todatetime(df):
        for col in df.columns:
            if "time" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

if __name__ == "__main__":
    from time import perf_counter as timer
    dataset = GLIDER()
    total = 0
    for i in range(len(dataset)):
        start_time = timer()
        data = dataset[i]
        end_time =  timer()
        dur = end_time - start_time
        total += dur
        print(f"{i} It took {(end_time - start_time)} seconds to load sample {i} average {total / (i+1)}")
        print(data)
        print(data.label_roll())
        print(data.binary())