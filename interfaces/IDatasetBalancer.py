#!/usr/bin/env python3
import abc
from typing import Iterable, Mapping

class IDatasetBalancer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eval_only_indeces() -> Iterable[int]:
        """Set the dataset to eval mode, the dataset ouput will not be balanced."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_indeces() -> Iterable[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def label_distributions(self) -> Mapping[str, Iterable[int]]:
        """Return a dictionary of label_keys and a list of indeces at which the corresponding labels are present in the dataset.
        
        For example, with the following labels: ["dog", "cat", "cat", "NA"]
        The output dictionary would be: {"dog": [0], "cat": [1, 2], "NA": [3]}
        
        Returns:
            Mapping[str, Iterable[int]]: {label: indeces, ...}
        """
        raise NotImplementedError

    def shuffle(self) -> None:
        raise NotImplementedError