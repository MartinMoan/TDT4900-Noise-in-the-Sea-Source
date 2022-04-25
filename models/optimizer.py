#!/usr/bin/env python3
import multiprocessing
import sys
import pathlib
from typing import Mapping, Type, Tuple

import torch
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IOptimizerProvider

class AdamaxProvider(IOptimizerProvider):
    def __init__(self, lr: float, weight_decay: float):
        self.lr = lr
        self.weight_decay = weight_decay

    def provide(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adamax(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @property
    def properties(self) -> Mapping[str, any]:
        return {"optimizer": str(torch.optim.Adamax), "lr": self.lr, "weight_decay": self.weight_decay}

class GeneralOptimizerProvider(IOptimizerProvider):
    def __init__(
        self, 
        optimizer_type: Type[torch.optim.Optimizer], 
        optimizer_args: Tuple[any, ...] = (), 
        optimizer_kwargs: Mapping[str, any] = {}):
        self.optimizer_type = optimizer_type
        self.optimizer_args = optimizer_args
        self.optimizer_kwargs = optimizer_kwargs

    def provide(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self.optimizer_type(model.parameters(), *self.optimizer_args, **self.optimizer_kwargs)
        
    @property
    def properties(self) -> Mapping[str, any]:
        return {"optimizer": self.optimizer_type, "args": self.optimizer_args, "kwargs": self.optimizer_kwargs}