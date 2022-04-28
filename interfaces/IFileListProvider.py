#!/usr/bin/env python3
import abc
from typing import Union, Iterable
import pathlib

class IFileListProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def list(self) -> Iterable[pathlib.Path]:
        raise NotImplementedError
