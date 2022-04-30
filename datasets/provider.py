#!/usr/bin/env python3
import multiprocessing
import sys
import pathlib
from typing import Mapping, Union

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IDatasetProvider, ITensorAudioDataset, ICustomDataset, IDatasetBalancer, IFeatureAccessor, ILabelAccessor, ILoggerFactory
from datasets.limiting import DatasetLimiter
from datasets.tensordataset import TensorAudioDataset

class BasicDatasetProvider(IDatasetProvider):
    def __init__(self, dataset: ITensorAudioDataset) -> None:
        self.dataset = dataset
    
    def provide(self) -> ITensorAudioDataset:
        return self.dataset

    @property
    def properties(self) -> Mapping[str, any]:
        return {}

class VerificationDatasetProvider(IDatasetProvider):
    def __init__(
        self, 
        clipped_dataset: ICustomDataset,
        limit: Union[float, int],
        balancer: IDatasetBalancer,
        randomize: bool,
        balanced: bool,
        feature_accessor: IFeatureAccessor,
        label_accessor: ILabelAccessor,
        logger_factory: ILoggerFactory) -> None:

        self.limit = limit
        self.feature_accessor = feature_accessor
        self.label_accessor = label_accessor
        self.balanced = balanced
        self.randomize = randomize
        self.balancer = balancer
        self.dataset = clipped_dataset

        self.limited = DatasetLimiter(
            dataset=clipped_dataset,
            limit=limit,
            balancer=balancer,
            randomize=randomize,
            balanced=balanced
        )

        self.tensordataset = TensorAudioDataset(
            dataset=self.limited,
            label_accessor=label_accessor,
            feature_accessor=feature_accessor,
            logger_factory=logger_factory
        )
        
    def provide(self) -> ITensorAudioDataset:
        return self.tensordataset

    @property
    def properties(self) -> Mapping[str, any]:
        out = {
            "limit": self.limit,
            "feature_accessor": self.feature_accessor,
            "label_accessor": self.label_accessor,
            "balanced": self.balanced,
            "randomize": self.randomize,
            "balancer": self.balancer
        }
        return out
