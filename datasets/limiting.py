#!/usr/bin/env python3
import sys
import pathlib
from typing import Union, Mapping, Iterable

import git
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from audiodata import LabeledAudioData
from clipping import ClippedDataset
from ICustomDataset import ICustomDataset
from ITensorAudioDataset import FileLengthTensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor, ITensorAudioDataset
from IMetricComputer import BinaryMetricComputer
from IDatasetBalancer import BalancedKFolder, DatasetBalancer
from ASTWrapper import ASTWrapper

class DatasetLimiter(ICustomDataset):
    def __init__(self, dataset: ICustomDataset, limit: Union[float, int], randomize: bool = False, balanced=True) -> None:
        self._dataset = dataset
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
            balancer = DatasetBalancer(dataset)
            splits = balancer._split_labels
            num_classes = len(splits.keys()) # both, neither, anthropogenic, biophonic
            n_per_class = int(num_in_subset / num_classes)
            class_numbers = {key: n_per_class for key in splits.keys()}
            
            if num_in_subset % num_classes != 0:
                diff = int(num_in_subset % num_classes)
                while diff > 0:
                    for key in class_numbers.keys():
                        class_numbers[key] += 1
                        diff -= 1
                        if diff <= 0:
                            break
                    
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

if __name__ == "__main__":
    clipped = ClippedDataset()