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
from scipy.io import wavfile
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
    def __init__(self, clip_duration_seconds: float = 10.0, verbose = False, suppress_warnings = None):
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
        self._pad = True
        self._clip_duration_seconds = clip_duration_seconds
        self._file_duration_seconds = self._audiofiles.num_samples.max() / self._audiofiles.sampling_rate.max()
        self._num_clips = int((len(self._audiofiles) * self._file_duration_seconds) / self._clip_duration_seconds)
        self._verbose = verbose
        self._suppress_warnings = config.ENV == "env" if suppress_warnings is None else suppress_warnings


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

    def _getitem_async(self, clip_index):
        clips_per_file = self._num_clips / len(self._audiofiles)
        file_index = int(clip_index / clips_per_file)

        file = self._audiofiles.iloc[file_index]
        first_clip_in_file_clip_index = file_index * clips_per_file
        nth_clip_in_file = clip_index - first_clip_in_file_clip_index
        clip_start_offset_seconds = nth_clip_in_file * self._clip_duration_seconds
        offset_td = timedelta(seconds=clip_start_offset_seconds)
        file_duration_seconds = (file.end_time - file.start_time).seconds
        with ThreadPool(processes=2) as pool:
            filepath = file.filename
            samples, sr = np.array([]), file.sampling_rate

            start_time = file.start_time + offset_td 
            end_time = start_time + timedelta(seconds=self._clip_duration_seconds)
            labels = self._get_labels(start_time, end_time)

            labels_task = pool.apply_async(self._get_labels, (start_time, end_time))
            expected_samples_shape = (int(self._clip_duration_seconds * self._audiofiles.sampling_rate.max()),)
            if config.VIRTUAL_DATASET_LOADING:
                samples, sr = np.zeros(expected_samples_shape), self._audiofiles.sampling_rate.max()
            elif clip_start_offset_seconds >= file_duration_seconds:
                # if not self._suppress_warnings:
                #     warnings.warn(f"The {clip_index}-th audio clip could not be read due to the parent audiofile having unexpected length. Returning zeroes with correct shape. {filepath}")
                # samples, sr = np.zeros(expected_samples_shape), self._audiofiles.sampling_rate.max()
                return self._getitem_async(clip_index+1) # TODO: This one does not spar joy...
            else:
                dur_to_read = min(self._clip_duration_seconds, (file.end_time - start_time).seconds)
                if dur_to_read > (file.end_time - start_time).seconds:
                    dur_to_read = (file.end_time - start_time).seconds

                loading_task = pool.apply_async(librosa.load, (filepath, ), {"sr": None, "offset": clip_start_offset_seconds, "duration": dur_to_read})
                samples, sr = loading_task.get()
                if len(samples) == 0 and not self._suppress_warnings:
                    warnings.warn(f"librosa.load returned an 0 samples for audiofile {filepath}. Reading parameters: offset: {clip_start_offset_seconds} duration: {dur_to_read} file duration seconds: {file_duration_seconds}")
                if len(samples) != expected_samples_shape[0]:
                    to_pad = np.zeros(expected_samples_shape[0] - len(samples))
                    samples = np.concatenate((samples, to_pad), axis=0)

            labels = labels_task.get()
            
            return LabeledAudioData(
                clip_index,
                filepath = filepath,
                num_channels = file.num_channels,
                sampling_rate = sr,
                start_time = start_time,
                end_time = end_time,
                samples = samples,
                labels = labels,
                labels_dict = self.classes()
            )

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
        audio_data = self._getitem_async(index)
        if self._verbose:
            print(f"{index} / {len(self)}", audio_data)
            
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

if __name__ == "__main__":
    dataset = GLIDER(clip_duration_seconds=10.0, verbose=True, suppress_warnings=True)
    
    for i in range(len(dataset)):
        data = dataset[i]

    