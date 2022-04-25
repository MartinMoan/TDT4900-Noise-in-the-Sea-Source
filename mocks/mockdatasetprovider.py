#!/usr/bin/env python3
import sys
import pathlib
from turtle import forward
from typing import Iterable, Mapping

import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IDatasetProvider, ITensorAudioDataset

class MockDatasetProvider(IDatasetProvider):
    def __init__(self, dataset: ITensorAudioDataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def provide(self) -> ITensorAudioDataset:
        return self.dataset

    @property
    def properties(self) -> Mapping[str, any]:
        return {f"{self.__class__.__name__}": "MOCK_DATASET_PROVIDER"}