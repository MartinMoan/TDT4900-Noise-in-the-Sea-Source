#!/usr/bin/env python3
import abc
from typing import Iterable

class IDatasetBalancer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eval_only_indeces() -> Iterable[int]:
        """Set the dataset to eval mode, the dataset ouput will not be balanced."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_indeces() -> Iterable[int]:
        raise NotImplementedError