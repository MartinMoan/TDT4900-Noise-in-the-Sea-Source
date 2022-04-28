#!/usr/bin/env python3
import abc
import pathlib

from typing import Mapping, Union

class ITracker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(trackables: Mapping[str, any]) -> None:
        raise NotImplementedError