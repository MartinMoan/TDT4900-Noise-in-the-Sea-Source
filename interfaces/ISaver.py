#!/usr/bin/env python3
import abc
import pathlib

import torch

class ISaver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save(self, model: torch.nn.Module, **kwargs) -> pathlib.Path:
        raise NotImplementedError