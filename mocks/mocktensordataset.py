#!/usr/bin/env python3
import sys
import pathlib
from typing import Iterable, Mapping

import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ITensorAudioDataset, ILoggerFactory
from datasets.glider.audiodata import AudioData, LabeledAudioData

class MockTensorDataset(ITensorAudioDataset):
    def __init__(
        self, 
        size: int, 
        label_shape: Iterable[int],
        feature_shape: Iterable[int],
        classes: Mapping[str, int],
        logger_factory: ILoggerFactory):
        
        self.size = size
        self._label_shape = label_shape
        self._feature_shape = feature_shape
        self._classes = classes
        self.logger = logger_factory.create_logger()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.log(f"{self.__class__.__name__} {index} / {len(self)}")
        features = torch.rand(*self._feature_shape)
        labels = torch.randint(0, 2, self._label_shape)
        return features, labels

    def example_shape(self) -> tuple[int, ...]:
        return self._feature_shape
    
    def label_shape(self) -> tuple[int, ...]:
        return self._label_shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size: {self.size}, label_shape: {self._label_shape}, feature_shape: {self._feature_shape})"

    def audiodata(self, index: int) -> AudioData:
        return super().audiodata(index)

    def classes(self) -> Mapping[str, int]:
        return self._classes

    def audiodata(self, index: int) -> LabeledAudioData:
        return super().audiodata(index)