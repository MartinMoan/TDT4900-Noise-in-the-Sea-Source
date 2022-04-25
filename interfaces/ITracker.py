#!/usr/bin/env python3
import abc
import pathlib

from typing import Mapping, Union

class ITracker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(
        metrics: Mapping[str, float], 
        model: str, 
        model_parameters_path: Union[str, pathlib.PosixPath], 
        *args,
        **kwargs) -> None:

        raise NotImplementedError