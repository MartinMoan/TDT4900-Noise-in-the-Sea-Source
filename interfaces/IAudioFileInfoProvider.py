#!/usr/bin/env python3
import abc
import pathlib

import pandas

class IAudioFileInfoProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def files(self) -> pandas.DataFrame:
        raise NotImplementedError