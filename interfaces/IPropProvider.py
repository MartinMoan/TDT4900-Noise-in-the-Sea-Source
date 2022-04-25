#!/usr/bin/env python3
import abc
from typing import Mapping

class IPropProvider(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def properties(self) -> Mapping[str, any]:
        raise NotImplementedError