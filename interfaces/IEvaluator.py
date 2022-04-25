#!/usr/bin/env python3
import abc
from typing import Tuple, Iterable

import torch

from .ITensorAudioDataset import ITensorAudioDataset
from .IPropProvider import IPropProvider

class IEvaluator(IPropProvider, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self, 
        model: torch.nn.Module, 
        dataset_indeces: Iterable[int], 
        dataset: ITensorAudioDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate model on the dataset indeces
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (truth, predictions)
        """
        raise NotImplementedError