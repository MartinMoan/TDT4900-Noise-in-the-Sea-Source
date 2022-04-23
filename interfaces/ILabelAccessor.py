#!/usr/bin/env python3
import abc
from typing import Union

import torch
import numpy as np

from datasets.glider.audiodata import LabeledAudioData

class ILabelAccessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, audio_data: LabeledAudioData, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError