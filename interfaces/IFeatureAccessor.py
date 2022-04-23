#!/usr/bin/env python3
import abc
from typing import Union

import torch
import numpy as np

from datasets.glider.audiodata import LabeledAudioData

class IFeatureAccessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, audio_data: LabeledAudioData) -> torch.Tensor:
        raise NotImplementedError

    def _to_single_channel_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(1, *tensor.shape)