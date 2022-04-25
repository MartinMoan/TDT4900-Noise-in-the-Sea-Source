#!/usr/bin/env python3
import multiprocessing
import sys
import pathlib
from typing import Mapping

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