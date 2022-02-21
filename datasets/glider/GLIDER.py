#!/usr/bin/env python3
from dbm import dumb
import pathlib
import sys
from datetime import timedelta

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

class GLIDERConfig:
    DURATION_SECONDS_PER_EXAMPLE = 60 * 20
    _SR = 128000
    _N_SAMPLES_PER_FILE = 76800000
    N_SAMPLES_PER_EXAMPLE = _SR * DURATION_SECONDS_PER_EXAMPLE
    

class GLIDER(BasicDataset):
    def __init__(self):
        self._labels = GLIDER._todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
        self._audiofiles = GLIDER._todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))
        self._non_label_columns = self._audiofiles.columns
        self._label_columns = self._labels.source_class.unique()
        
        self._audiofiles = self._audiofiles[(self._audiofiles.start_time >= self._labels.start_time.min())] # TODO: REMOVE ME!!
        
        self._label_audiofiles()
        if config.VIRTUAL_DATASET_LOADING:
            warnings.warn("The environment variable VIRTUAL_DATASET_LOADING is set to True, meaning that the GLIDER dataset loading class will only simulate loading datasets from disk. Ensure that this variable is not set during training or inference, as it is only intended to be used during local development.")
        self._audiofiles.sort_values(by=["start_time"], inplace=True, ascending=True)

    def _label_audiofiles(self):
        for idx in self._audiofiles.index:
            row = self._audiofiles.loc[idx]
            file_labels = self._labels[(self._labels.start_time >= row.start_time) & (self._labels.end_time <= row.end_time)]
            for label in file_labels.source_class.values:
                self._audiofiles.loc[idx, label] = config.POSITIVE_INSTANCE_CLASS_LABEL
        
        for col in self._label_columns:
            self._audiofiles[col] = self._audiofiles[col].fillna(config.NEGATIVE_INSTANCE_CLASS_LABEL)

    def _file_with_sample_n(self, n):
        cumulative_sum = 0
        for index in self._audiofiles.index:
            row = self._audiofiles.loc[index]
            if (cumulative_sum + row.num_samples) >= n:
                remainder = (cumulative_sum + row.num_samples) - n # How many samples back from the end of file at "index" can I find sample N
                before = row.num_samples - remainder # How many samples into file at "index" can I find sample N
                return index, before, remainder
            cumulative_sum += row.num_samples
        return -1, -1, -1

    def _get_files(self, start_sample_idx, num_samples):
        start, start_before, start_remainder = self._file_with_sample_n(start_sample_idx)
        end, end_before, end_remainder = self._file_with_sample_n(start_sample_idx + num_samples)
        return self._audiofiles.loc[start:end], start_before, end_remainder

    def _label_helper(self, labels_start, labels_end):
        print(labels_start, labels_end)
        subset = self._labels[(self._labels.start_time <= labels_end) & (self._labels.end_time >= labels_start)]
        first_label = subset[(subset.start_time == subset.start_time.min())]
        last_label = subset[(subset.end_time == subset.end_time.max())]
        print(subset)
        print()

        # print(self._labels[(self._labels.start_time == last_label.start_time) & (self._labels.end_time == last_label.end_time)])

    def __getitem__(self, index):
        n_samples_before_example = GLIDERConfig.N_SAMPLES_PER_EXAMPLE * index
        files, start_before, end_remainder = self._get_files(n_samples_before_example, GLIDERConfig.N_SAMPLES_PER_EXAMPLE)
        all_samples = []
        sr = None
        for index in files.index:
            file = files.loc[index]
            samples, sr = librosa.load(config.DATASET_DIRECTORY.joinpath(file.filename), sr=None)
            
            if sr != GLIDERConfig._SR:
                raise Exception(f"The sampling rate of file {file.filename} has unexpected value. Expected {GLIDERConfig._SR} but found {sr}")
            all_samples = np.concatenate((all_samples, samples))
        selected_samples = all_samples[start_before:-end_remainder]
        # self._label_helper(index, files.iloc[0], files.iloc[len(files) - 1], start_before, end_remainder)

        start_file = files.iloc[0]
        end_file = files.iloc[len(files) - 1]
        
        start_file_duration = start_file.end_time - start_file.start_time
        start_sample_percentage = start_before / start_file.num_samples
        into_start_file_delta = timedelta(seconds=start_file_duration.total_seconds() * start_sample_percentage)
        start_time = start_file.start_time + into_start_file_delta # the datetime of when the first sample of the selected sample occurs

        end_file_duration = end_file.end_time - end_file.start_time
        end_file_start_sample_percentage = (end_file.num_samples - end_remainder) / end_file.num_samples
        into_end_file_delta = timedelta(seconds=end_file_duration.total_seconds() * end_file_start_sample_percentage)
        end_time = end_file.start_time + into_end_file_delta
        self._label_helper(start_time, end_time)
        return selected_samples, sr
            
    def __len__(self):
        # if config.VIRTUAL_DATASET_LOADING:
        #     return 42
        # return len(self._audiofiles)
        return int(self._audiofiles.num_samples.sum() / GLIDERConfig.N_SAMPLES_PER_EXAMPLE)

    def classes(self):
        return {self._label_columns[idx]: idx for idx in range(len(self._label_columns))}

    def _todatetime(df):
        for col in df.columns:
            if "time" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df


if __name__ == "__main__":
    dataset = GLIDER()
    
    for i in range(len(dataset)):
        dataset[i]