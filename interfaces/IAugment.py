#!/usr/bin/env python3
import abc
from typing import List, Tuple

import torch

class IAugment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, data: torch.Tensor, label: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def __call__(self, data: torch.Tensor, label: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(data, label)

    @abc.abstractmethod
    def branching(self) -> int:
        """Return the branching factor of the Augment. The augment takes a single input tensor, and returns a set of agumented versions of the tensor. 
        The branching factor tells us the number of versions the augment will yield for a single input.

        Returns:
            int: the number of output tensors per input tensor
        """
        raise NotImplementedError