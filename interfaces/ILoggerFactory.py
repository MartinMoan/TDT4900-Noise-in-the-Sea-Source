#!/usr/bin/env python3
import abc
from typing import Type

from interfaces.ILogger import ILogger

class ILoggerFactory(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def logger_type(self) -> Type[ILogger]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_logger(self) -> ILogger:
        raise NotImplementedError