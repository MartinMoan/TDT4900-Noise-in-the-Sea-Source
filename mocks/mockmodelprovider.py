#!/usr/bin/env python3
import sys
import pathlib
from turtle import forward
from typing import Iterable, Mapping
import math

import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IModelProvider

class MockModel(torch.nn.Module):
    def __init__(self, output_shape: Iterable[int]) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.l1 = torch.nn.Linear(3, 2) # Such that MockModel.parameters() will return some parameters for optimizer

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[0]
        return torch.rand((batch_size, *self.output_shape), requires_grad=True) # requires grad so lossfuction/optimizer can work with the output

class MockModelProvider(IModelProvider):
    def __init__(self, output_shape: Iterable[int]) -> None:
        super().__init__()
        self.output_shape = output_shape

    def instantiate(self) -> torch.nn.Module:
        return MockModel(output_shape=self.output_shape)

    @property
    def properties(self) -> Mapping[str, any]:
        return {f"provider": "{self.__class__.__name__}"}