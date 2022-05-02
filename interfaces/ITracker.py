#!/usr/bin/env python3
import abc
import pathlib

from typing import Mapping, Union

from .ITensorAudioDataset import ITensorAudioDataset

class ITracker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(trackables: Mapping[str, any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def track_dataset(dataset: ITensorAudioDataset) -> None:
        raise NotImplementedError