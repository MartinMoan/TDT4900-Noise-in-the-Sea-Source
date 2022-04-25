#!/usr/bin/env python3
import abc
from typing import Mapping

from .ITensorAudioDataset import ITensorAudioDataset
from .IPropProvider import IPropProvider

class IDatasetProvider(IPropProvider, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def provide(self) -> ITensorAudioDataset:
        raise NotImplementedError