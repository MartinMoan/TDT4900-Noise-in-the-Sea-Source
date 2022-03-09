#!/usr/bin/env python3
import pathlib
import sys
from multiprocessing.pool import ThreadPool, Pool
from datetime import timedelta
from dataclasses import dataclass

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
        print(self._audiofiles.filename)
        local_audiofiles = config.list_local_audiofiles()
        local_audiofiles = [str(path) for path in local_audiofiles]
        print(local_audiofiles[:10])
        print(self._audiofiles.filename.values[:10])
        print(self._audiofiles.filename.isin(local_audiofiles))

        self._audiofiles = self._audiofiles.loc[self._audiofiles.filename.isin(local_audiofiles)]
        print(self._audiofiles)
        
        self._label_columns = self._labels.source_class.unique()
        self._classdict = self.classes()
        self._label_audiofiles()
        if config.VIRTUAL_DATASET_LOADING:
            warnings.warn("The environment variable VIRTUAL_DATASET_LOADING is set to True, meaning that the GLIDER dataset loading class will only simulate loading datasets from disk. Ensure that this variable is not set during training or inference, as it is only intended to be used during local development.")
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
        samples, sr = librosa.load(file.filename, sr=None)
        return LabeledAudioData(
            index,
            pathlib.Path(file.filename), 
            file.num_channels, 
            file.sampling_rate, 
            file.num_samples, 
            file.start_time, 
            file.end_time,
            samples, 
            labels,
            self._classdict)

    def _getitem_async(self, index):
        file = self._audiofiles.iloc[index]
        print(file)
        with ThreadPool(processes=2) as pool:
            async_samples_result = pool.apply_async(librosa.load, (file.filename, ), {"sr": None})
            async_labels_result = pool.apply_async(self._get_labels, (file, ))

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
        print(data.fill_labels())