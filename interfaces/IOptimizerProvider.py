#!/usr/bin/env python3
import abc

import torch

from .IPropProvider import IPropProvider

class IOptimizerProvider(IPropProvider, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def provide(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        raise NotImplementedError
