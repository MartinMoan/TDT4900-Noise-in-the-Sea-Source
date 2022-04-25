#!/usr/bin/env python3
import abc

import sklearn.model_selection

from .IDatasetVerifier import IDatasetVerifier
from .IModelProvider import IModelProvider
from .IDatasetProvider import IDatasetProvider
from .ITracker import ITracker
from .IFolder import IFolder

class ICrossEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def kfoldcv(self) -> None:
        raise NotImplementedError

    @property
    def model_provider(self) -> IModelProvider:
        raise NotImplementedError

    @property
    def dataset_provider(self) -> IDatasetProvider:
        raise NotImplementedError
    
    @property
    def tracker(self) -> ITracker:
        raise NotImplementedError

    @property
    def folder(self) -> IFolder:
        return self.folder