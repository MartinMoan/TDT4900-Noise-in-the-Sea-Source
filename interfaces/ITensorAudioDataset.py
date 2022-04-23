#!/usr/bin/env python3
import abc
from typing import  Mapping

import torch

from datasets.glider.audiodata import AudioData

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

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def audiodata(self, index: int) -> AudioData:
        raise NotImplementedError