#!/usr/bin/env python3
import abc
from typing import Tuple

import torch

class IScaler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, index: int, X: torch.Tensor, Y: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        raise NotImplementedError