#!/usr/bin/env python3
import abc
from typing import Mapping

import torch

from .IPropProvider import IPropProvider

class IModelProvider(IPropProvider, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def instantiate(self) -> torch.nn.Module:
        raise NotImplementedError
