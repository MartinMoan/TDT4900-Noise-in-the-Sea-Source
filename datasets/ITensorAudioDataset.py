#!/usr/bin/env python3
import sys, pathlib, abc
from tkinter import Label
from typing import Mapping, Iterable, Union
import git
import torch
import numpy as np
import librosa

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from ICustomDataset import ICustomDataset
from glider.audiodata import LabeledAudioData
from GLIDER import GLIDER 
import warnings

def _to_tensor(nparray: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.array(nparray), dtype=torch.float32, requires_grad=False)

class ILabelAccessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, audio_data: LabeledAudioData, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class IFeatureAccessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, audio_data: LabeledAudioData) -> torch.Tensor:
        raise NotImplementedError

    def _to_single_channel_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(1, *tensor.shape)

class ITensorAudioDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__getitem__") and callable(subclass.__getitem__) and 
            hasattr(subclass, "__len__") and callable(subclass.__len__) and
            hasattr(subclass, "classes") and callable(subclass.classes) and 
            hasattr(subclass, "example_shape") and callable(subclass.example_shape) or
            NotImplemented
        )

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the X, Y (input, truth) pytorch Tensors"""
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get the length of the audio dataset"""
        raise NotImplementedError

    @abc.abstractmethod
    def classes(self) -> Mapping[str, int]:
        """Get a dict of with classname: index pairs"""
        raise NotImplementedError

    @abc.abstractmethod
    def example_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def label_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

#####
##### Implementations:
#####

class LabelRollAccessor(ILabelAccessor):
    def __call__(self, audio_data: LabeledAudioData, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # return _to_tensor(audio_data.label_roll())
        return super().__call__(audio_data, features)

class BinaryLabelAccessor(ILabelAccessor):
    def __call__(self, audio_data: LabeledAudioData, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return _to_tensor(audio_data.binary())

class MelSpectrogramFeatureAccessor(IFeatureAccessor):
    def __init__(self, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512) -> None:
        self._n_mels = n_mels
        self._n_fft = n_fft
        self._hop_length = hop_length

    def __call__(self, audio_data: LabeledAudioData) -> torch.Tensor:    
        samples, sr = audio_data.samples, audio_data.sampling_rate
        if config.VIRTUAL_DATASET_LOADING:
            output_shape = (self._n_mels, 1 + int(len(samples) / self._hop_length))
            return torch.zeros((1, *output_shape), dtype=torch.float32, requires_grad=False)

        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=self._n_mels, n_fft=self._n_fft, hop_length=self._hop_length))
        S_db = np.flip(S_db, axis=0)
        
        # Normalize across frequency bands (give every frequency band/bin 0 mean and unit variance)
        # TODO: Handle exceptions where there is no variance within any freqency bin. E.g. "denominator" bellow is np.nan along one or more frequencies
        mean = np.mean(S_db, axis=1).reshape((-1, 1))
        numerator = (S_db - mean)
        denominator = np.std(S_db, axis=1).reshape((-1, 1))
        S_db = numerator * (1/denominator)
        return self._to_single_channel_batch(_to_tensor(S_db))

class FileLengthTensorAudioDataset(ITensorAudioDataset):
    def __init__(self, dataset: ICustomDataset, label_accessor: ILabelAccessor, feature_accessor: IFeatureAccessor) -> None:
        if not isinstance(dataset, ICustomDataset):
            raise TypeError(f"Argument dataset has invalid type. Expected {ICustomDataset} but received {type(dataset)}")
        if not isinstance(label_accessor, ILabelAccessor):
            raise TypeError(f"Argument label_accessor has invalid type. Expected {ILabelAccessor} but received {type(label_accessor)}")
        if not isinstance(feature_accessor, IFeatureAccessor):
            raise TypeError(f"Argument feature_accessor has invalid type. Expected {IFeatureAccessor} but received {type(feature_accessor)}")

        self._dataset = dataset
        self._label_accessor = label_accessor
        self._feature_accessor = feature_accessor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_data = self._dataset[index]
        features = torch.nan_to_num(self._feature_accessor(audio_data), nan=0, posinf=10, neginf=-10)
        labels = self._label_accessor(audio_data, features)
        return index, features, labels

    def __len__(self) -> int:
        return len(self._dataset)

    def classes(self) -> Mapping[str, int]:
        return self._dataset.classes()

    def example_shape(self) -> tuple[int, ...]:
        if len(self) > 0:
            audio_data = self._dataset[0]
            features = self._feature_accessor(audio_data)
            return features.shape
        else:
            warnings.warn("The TensorAudioDataset has length 0, this will likely cause unexpected results")
            return None

    def label_shape(self) -> tuple[int, ...]:
        if len(self) > 0:
            audio_data = self._dataset[0]
            features = self._feature_accessor(audio_data)
            labels = self._label_accessor(audio_data, features)
            return labels.shape
        else:
            warnings.warn("The TensorAudioDataset has length 0, this will likely cause unexpected results")
            return None

if __name__ == "__main__":
    dataset = FileLengthTensorAudioDataset(dataset=GLIDER(clip_duration_seconds = 10.0), label_accessor = BinaryLabelAccessor(), feature_accessor = MelSpectrogramFeatureAccessor())
    _indeces = [40414, 146869, 78997, 162159, 174450, 75375, 80172, 11896, 45205, 212519, 75177, 228142, 88527, 128200, 153709, 117738, 50659, 10586, 122117, 180314, 81489, 58191, 94471, 82012, 199068, 244187, 232152, 233318, 23947, 182991, 635, 215504, 64169, 226989, 12302, 136440, 244239, 28445, 46475, 120555, 80150, 163527, 246924, 135159, 188942, 228160, 106653, 36583, 53382, 34099, 36762, 146038, 83628, 140742, 231528, 67522, 93338, 248063, 87903, 113978, 55655, 88584, 126586, 131694]
    indeces = [idx for idx in _indeces if idx < len(dataset)]
    print(len(dataset))
    for index in indeces:
        print(dataset[index])