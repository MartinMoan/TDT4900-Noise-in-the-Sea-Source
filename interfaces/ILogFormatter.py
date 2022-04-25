#!/usr/bin/env python3
import abc

from typing import Iterable, Tuple

class ILogFormatter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def header(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def format(self, *args, **kwargs) -> Iterable[str]:
        raise NotImplementedError
