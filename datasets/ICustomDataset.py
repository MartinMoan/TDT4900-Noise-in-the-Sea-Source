#!/usr/bin/env python3
import abc
from typing import Iterable, Mapping
import sys
import torch
from glider.audiodata import LabeledAudioData


class ICustomDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__getitem__") and callable(subclass.__getitem__) and 
            hasattr(subclass, "__len__") and callable(subclass.__len__) and
            hasattr(subclass, "classes") and callable(subclass.classes) and 
            hasattr(subclass, "example_shape") and callable(subclass.example_shape) or
            hasattr(subclass, "example_shapes") and callable(subclass.example_shapes) or
            NotImplemented
        )

    @abc.abstractmethod
    def __getitem__(self, index: int) -> LabeledAudioData:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def classes(self) -> Mapping[str, int]:
        raise NotImplementedError

    @abc.abstractmethod
    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError