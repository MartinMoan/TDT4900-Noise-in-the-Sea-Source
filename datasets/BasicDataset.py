#!/usr/bin/env python3
import sys
import torch
from glider.audiodata import LabeledAudioData

class BasicDataset(torch.utils.data.Dataset):
    def _not_implemented_error_(self, fname):
        raise NotImplementedError(f"The method {self.__class__.__name__}.{fname} method was called. The inheriting class should implement should override this method")

    def __getitem__(self, index) -> LabeledAudioData:
        self._not_implemented_error_(sys._getframe().f_code.co_name)

    def __len__(self) -> int:
        self._not_implemented_error_(sys._getframe().f_code.co_name)

    def classes(self) -> dict:
        self._not_implemented_error_(sys._getframe().f_code.co_name)