#!/usr/bin/env python3
import abc

from .IScaler import IScaler
from .ITensorAudioDataset import ITensorAudioDataset

class ITransformer(IScaler, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, dataset: ITensorAudioDataset) -> IScaler:
        raise NotImplementedError