#!/usr/bin/env python3
import abc
from typing import Tuple, Mapping

from .ITensorAudioDataset import ITensorAudioDataset

class IDatasetVerifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def verify(self, dataset: ITensorAudioDataset) -> Tuple[bool, Mapping[str, any]]:
        raise NotImplementedError