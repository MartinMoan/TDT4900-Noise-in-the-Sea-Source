#!/usr/bin/env python3
from distutils.file_util import copy_file
import pathlib
import sys

import pandas as pd
import numpy as np
import torch
from rich import print
import librosa
import git
import warnings

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from BasicDataset import BasicDataset

class GLIDER(BasicDataset):
    def __init__(self):
        self._labels = GLIDER._todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
        self._audiofiles = GLIDER._todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))
        self._non_label_columns = self._audiofiles.columns
        self._label_columns = self._labels.source_class.unique()
        self._label_audiofiles()
        if config.VIRTUAL_DATASET_LOADING:
            warnings.warn("The environment variable VIRTUAL_DATASET_LOADING is set to True, meaning that the GLIDER dataset loading class will only simulate loading datasets from disk. Ensure that this variable is not set during training or inference, as it is only intended to be used during local development.")
        
    def _label_audiofiles(self):
        for idx in self._audiofiles.index:
            row = self._audiofiles.loc[idx]
            file_labels = self._labels[(self._labels.start_time >= row.start_time) & (self._labels.end_time <= row.end_time)]
            for label in file_labels.source_class.values:
                self._audiofiles.loc[idx, label] = config.POSITIVE_INSTANCE_CLASS_LABEL
        
        for col in self._label_columns:
            self._audiofiles[col] = self._audiofiles[col].fillna(config.NEGATIVE_INSTANCE_CLASS_LABEL)

    def _get_files_by_label(self, label):
        return self._audiofiles[(self._audiofiles[label] == config.POSITIVE_INSTANCE_CLASS_LABEL)]

    def _get_anthropogenic_files(self):
        return self._get_files_by_label("Anthropogenic")
    
    def _get_biophonic_files(self):
        return self._get_files_by_label("Biophonic")

    def __len__(self):
        if config.VIRTUAL_DATASET_LOADING:
            return 42
        return len(self._audiofiles)

    def _get_samples(self, filename):
        samples, sr = [], None
        if config.VIRTUAL_DATASET_LOADING:
            dur = 3
            sr = 200
            samples = np.random.random(dur * sr)
        else:
            samples, sr = librosa.load(filename, sr=None)
        return samples, sr

    def _get_labels(self, row):
        if config.VIRTUAL_DATASET_LOADING:
            return {col: np.random.choice([config.POSITIVE_INSTANCE_CLASS_LABEL, config.NEGATIVE_INSTANCE_CLASS_LABEL]) for col in self._label_columns}
        return {col: row[col] for col in self._label_columns}    

    def __getitem__(self, index):
        row = self._audiofiles.iloc[index]
        filename = config.DATASET_DIRECTORY.joinpath(row.filename)
        samples, sr = self._get_samples(filename)
        labels = self._get_labels(row)
        return filename, samples, sr, labels

    def classes(self):
        return {self._label_columns[idx]: idx for idx in range(len(self._label_columns))}

    def _todatetime(df):
        for col in df.columns:
            if "time" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df


if __name__ == "__main__":
    dataset = GLIDER()
    print(dataset[0])
