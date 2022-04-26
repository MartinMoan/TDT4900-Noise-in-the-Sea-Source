#!/usr/bin/env python3
import sys
import pathlib
from typing import Union, Mapping, Iterable, Type

import git
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from datasets.glider.audiodata import LabeledAudioData
from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from interfaces import ICustomDataset, IDatasetBalancer, ILoggerFactory
from tracking.loggerfactory import LoggerFactory
from tracking.logger import Logger, LogFormatter
from datasets.balancing import CachedDatasetBalancer, DatasetBalancer
from datasets.binjob import Binworker

class DatasetLimiter(ICustomDataset):
    def __init__(self, dataset: ICustomDataset, limit: Union[float, int], balancer: IDatasetBalancer, randomize: bool = False, balanced=True) -> None:
        self._dataset = dataset
        self.balancer = balancer
        self._dataset_indeces = self._get_subset_indeces(dataset, limit, randomize, balanced)

    def _parse_subset_size(self, limit: Union[float, int]) -> int:
        _num_in_subset = None
        
        if type(limit) == float:
            if limit > 1.0 or limit < 0.0:
                raise ValueError(limit)
            _num_in_subset = int(limit * len(self.dataset))
        elif type(limit) == int:
            _num_in_subset = limit
        else:
            raise TypeError

        return _num_in_subset

    def _get_subset_indeces(
        self, 
        dataset: ICustomDataset, 
        limit: Union[float, int], 
        randomize: bool = False,
        balance: bool = True) -> Iterable[int]:

        num_in_subset = self._parse_subset_size(limit)
        raw_indeces = np.arange(0, len(dataset))
        if balance:
            splits = self.balancer._split_labels
            num_classes = len(splits.keys()) # both, neither, anthropogenic, biophonic
            n_per_class = int(num_in_subset / num_classes)
            class_numbers = {key: n_per_class for key in splits.keys()}
            
            for key in class_numbers.keys():
                if class_numbers[key] > len(splits[key]):
                    excess = class_numbers[key] - len(splits[key])
                    class_numbers[key] = len(splits[key])
                    otherkeys = [otherkey for otherkey in splits.keys() if otherkey != key]
                    while excess > 0:
                        for otherkey in otherkeys:
                            class_numbers[otherkey] += 1
                            excess -= 1
                            if excess == 0:
                                break

            if randomize:
                indeces = []
                for key in splits.keys():
                    class_indeces = splits[key]
                    n_in_class = class_numbers[key]
                    indeces += list(np.random.choice(class_indeces, n_in_class, replace=False))
                np.random.shuffle(indeces)
                return indeces
            else:
                indeces = []
                for key in splits.keys():
                    indeces += splits[key][:n_per_class]
                return indeces
        if randomize:
            choices = np.random.choice(raw_indeces, num_in_subset, replace=False)
            return list(choices)
        else:
            return list(range(0, num_in_subset))

    def __len__(self) -> int:
        return len(self._dataset_indeces)

    def __getitem__(self, index) -> LabeledAudioData:
        if index > len(self):
            raise IndexError(f"index {index} out of range for {self.__class__.__name__} object with length {len(self)}")
        raw_index = self._dataset_indeces[index]
        return self._dataset[raw_index]

    def classes(self) -> Mapping[str, int]:
        return self._dataset.classes()

    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        return self._dataset.example_shapes()

    def __repr__(self):
        return f"{self.__class__.__name__}(_dataset={repr(self._dataset)}, _dataset_indeces={repr(self._dataset_indeces)})"

class ProportionalDatasetLimiter(ICustomDataset):
    def __init__(
        self, 
        dataset: ICustomDataset, 
        balancer: IDatasetBalancer,
        logger_factory: ILoggerFactory,
        size: Union[float, int]) -> None:
        super().__init__()

        self.dataset = dataset
        self.balancer = balancer
        self.logger = logger_factory.create_logger()

        if isinstance(size, float):
            if size < 0.0 or size > 1.0:
                raise ValueError
            self.scale = size # the percentage of each group/all label groups to draw
        elif isinstance(size, int):
            if size < 0:
                raise ValueError
            elif size > len(self.dataset):
                raise ValueError
            self.scale = size / len(self.dataset)
        else:
            raise TypeError
        
        distributions = balancer.label_distributions()
        """
            Example:
            distributions = {
                "anthropogenic": [0, 1, 2...], 
                "neither": [3, 5, ...],
                "both": [6, 9, 543, 66, ...],
                "biophonic": [...]
            }
        """
        self.scaled_distributions = {key: np.random.choice(distributions[key], size=round(len(distributions[key]) * self.scale)) for key in distributions.keys()}
        self.total_length = np.sum([len(idxs) for idxs in self.scaled_distributions.values()])
        for key, idxs in self.scaled_distributions.items():
            part_before = len(distributions[key]) / len(dataset)
            part_after = len(idxs) / self.total_length
            self.logger.log(f"Label distribution: '{key}' before scaling {len(distributions[key])} / {len(dataset)} ({part_before*100:.2f}%), after scaling {len(idxs)} / {self.total_length} ({part_after*100:.2f}%)")
        self.indeces = np.concatenate([indeces for indeces in self.scaled_distributions.values()], axis=0)

    def __getitem__(self, index: int) -> LabeledAudioData:
        return self.dataset[self.indeces[index]]

    def __len__(self) -> int:
        return len(self.indeces)

    def classes(self) -> Mapping[str, int]:
        return self.dataset.classes()
    
    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        return self.dataset.example_shapes()

    def as_dict(self):
        relevant_values = {
            "scaled_distributions": self.scaled_distributions,
            "total_length": self.total_length,
            "indeces": self.indeces,
            "dataset": self.dataset,
            "distributions": self.distributions
        }
        return relevant_values

    def __repr__(self):
        relevant_values = self.as_dict()
        out = f"{self.__class__.__name__}( {repr(relevant_values)} )"
        return repr(out)

    def __str__(self):
        relevant_values = self.as_dict()
        out = f"{self.__class__.__name__}( {str(relevant_values)} )"
        return str(out)

if __name__ == "__main__":
    logger_factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(LogFormatter(),)
    )
    worker = Binworker()

    clipped = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=worker,
        clip_duration_seconds=10.0,
        clip_overlap_seconds=4.0
    )
    balancer = CachedDatasetBalancer(
        dataset=clipped,
        logger_factory=logger_factory,
        worker=worker,
        force_recache=False
    )

    limited = ProportionalDatasetLimiter(
        dataset=clipped,
        balancer=balancer,
        logger_factory=logger_factory,
        size=20
    )

    print(len(limited), len(clipped), len(limited) / len(clipped))