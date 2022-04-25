#!/usr/bin/env python3
import abc
from typing import Iterable

import torch

from .ITensorAudioDataset import ITensorAudioDataset
from .IPropProvider import IPropProvider

class ITrainer(IPropProvider, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(
        self, 
        model: torch.nn.Module, 
        sample_indeces: Iterable[int], 
        dataset: ITensorAudioDataset
        ) -> torch.nn.Module:
        raise NotImplementedError