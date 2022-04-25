#!/usr/bin/env python3
import abc
from typing import Mapping, Generator, Iterable, Tuple

from .IPropProvider import IPropProvider

class IFolder(IPropProvider, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def split(self, iterable: Iterable[any]) -> Generator[Tuple[Iterable[int], Iterable[int]], None, None]:
        raise NotImplementedError
