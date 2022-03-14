from multiprocessing.spawn import import_main_path
import pathlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from rich import print

@dataclass
class AudioData:
    """Class representing an audiofile with samples"""
    _index: int
    filepath: pathlib.PosixPath # local filepath to the audiofile
    num_channels: int # number of channels of the audiofile
    sampling_rate: int # the sampling rate of the audiofile
    # num_samples: int # the number of samples of the audio recording
    start_time: np.datetime64 # the starting timestamp of the recording
    end_time: np.datetime64 # the ending timestamp of the recording
    
    samples: np.ndarray # the sample values of the recording

    @property
    def num_samples(self) -> int:
        return len(self.samples)
    

@dataclass
class LabeledAudioData(AudioData):
    """Class representing a labeled audio dataset example (e.g. an audio file/series of samles with associated labels)"""
    labels: pd.DataFrame # the overlaping labels of the example
    labels_dict: dict # the label dictionary with {str: int} with int being the vertical axis of the sample_labels representing the label str
    
    def label_roll(self, N=None) -> np.ndarray:
        """Generate the label roll consisting of N elements, e.g. a C, N matrix of 0/1 values, where C equals the number of classes, and N being a positive integer. 
        The label instances will be distributed over N accoring to the file start- and end timestamps along any overlapping label instances. 
        Args:
            N (None | int): The lenght of the output matrix second axis. If None the number of samples of the example is used.
        Returns:
            numpy.ndarray : the label roll with shape (C, N)
        """
        if N is None:
            N = self.num_samples

        num_classes = len(self.labels_dict)
        sample_labels = np.zeros((num_classes, N))
        for label_idx in self.labels.index:
            label = self.labels.loc[label_idx]
            s = label.start_time if label.start_time >= self.start_time else self.start_time
            e = label.end_time if label.end_time <= self.end_time else self.end_time
            first_sample = int((s - self.start_time) / (self.end_time - self.start_time) * N)
            last_sample = int((e - self.start_time) / (self.end_time - self.start_time) * N)
            sample_labels[self.labels_dict[label.source_class], first_sample:last_sample] = 1
        return sample_labels

    def binary(self) -> np.ndarray:
        """Get the binary (true/false) label instance vector. E.g.: vector of C values, where C is the number of classes in the dataset, with 1 representing a true instance of class C.

        Returns:
            numpy.ndarray : The binary label vector with shape (C, )
        """
        classes = self.labels.source_class.values
        output = np.zeros(len(self.labels_dict.keys()))
        for classname in classes:
            output[self.labels_dict[classname]] = 1
        return output