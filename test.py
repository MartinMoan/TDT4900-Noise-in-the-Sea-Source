#!/usr/bin/env python3
import sys
import pathlib
from typing import Tuple, Iterable, Mapping

import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from models import saver
from interfaces import IModelProvider, ILoggerFactory, IDatasetVerifier, ICrossEvaluator, ITracker, IDatasetProvider, ITrainer, IEvaluator, IMetricComputer, ITensorAudioDataset

class Test(IEvaluator):
    def evaluate(self, model: torch.nn.Module, dataset_indeces: Iterable[int], dataset: ITensorAudioDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().evaluate(model, dataset_indeces, dataset)

    @property
    def properties(self) -> Mapping[str, any]:
        return super().properties

if __name__ == "__main__":
    t = Test()
    
