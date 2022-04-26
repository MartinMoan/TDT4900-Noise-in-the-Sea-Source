#!/usr/bin/env python3
import abc
from typing import Iterable, Tuple, Mapping

class ITabularLogger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_row(self, row: Mapping[str, any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def format(self, order_by: Iterable[Tuple[str, str]] = [], col_order: Iterable[str] = []) -> None:
        raise NotImplementedError