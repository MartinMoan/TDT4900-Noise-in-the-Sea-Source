#!/usr/bin/env python3
import abc

class ILogger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log(self, *args, **kwargs) -> None:
        raise NotImplementedError
