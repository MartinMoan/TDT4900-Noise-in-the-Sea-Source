#!/usr/bin/env python3
import pathlib
import sys
import inspect
import copy
import warnings
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import librosa
from librosa import display
from librosa.core.audio import resample
from librosa.feature.spectral import melspectrogram
from librosa.filters import mel
import numpy as np
import torch
from rich import print

from dclde.dataset import DCLDE

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, featurefunc, verbose=False, resample=True):
        super().__init__()
        self._data = DCLDE(resample=resample)
        # self._classes = {
        #     "biophonic": 0,
        #     "anthropogenic": 1,
        #     "geogenic": 2,
        #     # 3: "noise"
        # }
        self._classes = {
            "biophonic": 0
        }
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
        return copy.deepcopy(self._classes)

    def __len__(self):
        return len(self._data)

    def _labels(self, species_annotations):
        classes = np.zeros(len(self._classes.keys()))
        if len(species_annotations) > 0:
            classes[self._classes["biophonic"]] = 1
        return classes

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
            filepath, samples, sr, species_annotations = self._data[index]
            
            # TODO: Fix implementation; naive approach to add zeros if file is not of expected 60 second length
            if len(samples) != int(sr * 60): # Expect 1 minute recording
                zeros = np.zeros(int(sr*60) - len(samples))
                samples = np.concatenate([samples, zeros])

            if self.verbose:
                print(filepath)
            
            features = self._feature(samples, sr)
            labels = self._labels(species_annotations)
            X, Y = self._to_single_channel_batch(self._to_tensor(features)), self._to_tensor(labels)
            X, Y = self._store_shapes(X, Y)
            return X, Y
        except Exception as ex:
            warnings.warn(f"An error occured when loading sample {index} with TorchDataset {str(ex)}. Returning zeroes with correct dimensions by default.")
            X = torch.zeros(self._seen_X_shape, dtype=torch.float32)
            Y = torch.zeros(self._seen_Y_shape, dtype=torch.float32)
            return X, Y

class MelSpectrogramDataset(TorchDataset):
    def __init__(self, n_mels=128, n_fft=2048, *args, **kwargs):
        self._n_mels = n_mels
        self._n_fft = n_fft
        super().__init__(featurefunc=self._melspect, *args, **kwargs)

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
    data = MelSpectrogramDataset(verbose=True, resample=True)

    X, Y = data[4]
    print(X.shape, Y)
