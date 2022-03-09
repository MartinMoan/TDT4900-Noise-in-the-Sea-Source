#!/usr/bin/env python3
from argparse import ArgumentError
from importlib.resources import path
import pathlib
import sys
import inspect
import copy
import warnings
import re

import matplotlib.pyplot as plt
import librosa
from librosa import display
from librosa.core.audio import resample
from librosa.feature.spectral import melspectrogram
from librosa.filters import mel
import numpy as np
import torch
from rich import print
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from BasicDataset import BasicDataset
        
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, basic_dataset, featurefunc=None, verbose=False, resample=True):
        super().__init__()
        if not isinstance(basic_dataset, BasicDataset):
            raise Exception(f"basic_dataset argument of incorrect type. Expected BasicDataset but received {type(basic_dataset)}")
        self._data = basic_dataset
        
        if featurefunc is None:
            raise Exception("Error: argument featurefunc is None. Must be not None callable with signature (samples, sr)")
        
        if not callable(featurefunc):
            raise Exception(f"Error: arugment featurefunc must be callable, but received uncallable of type {type(featurefunc)}")
            
        signature = dict(inspect.signature(featurefunc).parameters)
        if len(signature.keys()) != 2:
            raise Exception(f"Error: argument featurefunc has incorrect number of expected arguments, expected callable with arguments (samples, sr) but received callable with arguments {signature}")
        
        self._feature = featurefunc
        self.verbose = verbose

        self._seen_X_shape = None
        self._seen_Y_shape = None

    def classes(self):
        return copy.deepcopy(self._data.classes()) #copy.deepcopy(self._classes)

    def __len__(self):
        return len(self._data)

    def _labels(self, species_annotations):
        class_labels = self._data.classes()
        output = np.zeros(len(class_labels))
        for key in class_labels.keys():
            output[class_labels[key]] = species_annotations[key]
        return output

    def _to_tensor(self, nparray):
        return torch.tensor(np.array(nparray), dtype=torch.float32, requires_grad=False)
        
    def _to_single_channel_batch(self, tensor):
        """Convert an input tensor with shape (batch_size/-1, n_mels, N) to a single channel image with shape (batch_size/-1, 1, n_mels, N) such that it works with image based models

        Args:
            tensor (torch.tensor): Tensor with shape (-1, width, height)
        Returns:
            tensor (torch.tensor): Tensor with shape (-1, 1, width, height)
        """
        out = tensor.view(1, *tensor.shape)
        return out

    def _store_shapes(self, X, Y):
        if self._seen_X_shape is None:
            self._seen_X_shape = X.shape
        if self._seen_Y_shape is None:
            self._seen_Y_shape = Y.shape
        return X, Y

    def __getitem__(self, index):
        try:
            labeled_audio_data = self._data[index]
            sr = labeled_audio_data.sampling_rate
            filepath = labeled_audio_data.filepath
            samples = labeled_audio_data.samples
            
            # TODO: Fix implementation; naive approach to add zeros if file is not of expected 10 minute length
            if len(samples) != int(sr * 60 * 10): # Expect 10 minute recording
                zeros = np.zeros(int(sr * 60 * 10) - len(samples))
                samples = np.concatenate([samples, zeros])

            if self.verbose:
                print(index, str(filepath.name))

            features = self._feature(samples, sr)
            
            labels = labeled_audio_data.fill_labels()
            X, Y = self._to_single_channel_batch(self._to_tensor(features)), self._to_tensor(labels)
            X, Y = self._store_shapes(X, Y)
            return X, Y

        except Exception as ex:
            warnings.warn(f"An error occured when loading sample {index} with TorchDataset {str(ex)}. Returning zeroes with correct dimensions by default.")
            X = torch.zeros(self._seen_X_shape, dtype=torch.float32)
            Y = torch.zeros(self._seen_Y_shape, dtype=torch.float32)
            return X, Y

class MelSpectrogramDataset(AudioDataset):
    def __init__(self, *args, n_mels=128, n_fft=2048, **kwargs):
        self._n_mels = n_mels
        self._n_fft = n_fft
        super().__init__(*args, featurefunc=self._melspect, **kwargs)

    def _melspect(self, samples, sr):
        """Compute the mel_spectrogram of the input samples, and normalize frequency bins to have 0 mean and unit variance

        Args:
            samples (np.array with shape (N, 1) or (N, )): the sampled data
            sr (int > 0): the sample rate of the sampled data

        Returns:
            [np.array (n_mels, N)]: Standardized mel_spectrogram across frequency bins
        """
        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=self._n_mels, n_fft=self._n_fft))
        S_db = np.flip(S_db, axis=0)
        
        # Normalize across frequency bands (give every frequency band/bin 0 mean and unit variance)
        # TODO: Handle exceptions where there is no variance within any freqency bin. E.g. "denominator" bellow is np.nan along one or more frequencies
        mean = np.mean(S_db, axis=1).reshape((-1, 1))
        numerator = (S_db - mean)
        denominator = np.std(S_db, axis=1).reshape((-1, 1))
        S_db = numerator * (1/denominator)
        return S_db

if __name__ == "__main__":
    glider_path = pathlib.Path(__file__).parent.joinpath("glider")
    sys.path.insert(0, str(glider_path))
    from glider.GLIDER import GLIDER
    glider = GLIDER()
    data = MelSpectrogramDataset(glider, verbose=True, resample=True)
    
    X, Y = data[4]
    print(X.shape, Y)
