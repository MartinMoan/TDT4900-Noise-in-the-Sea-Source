#!/usr/bin/env python3
import abc
from typing import Iterable, Callable, Tuple, Mapping, Iterable

class IAsyncWorker(metaclass=abc.ABCMeta):
    def default_aggregation(results: Iterable[Iterable[any]]) -> Iterable[any]:
        output = []
        for iterable in results:
            for result in iterable:
                output.append(result)
        return output

    @abc.abstractmethod
    def apply(
        self, 
        iterable: Iterable[any], 
        function: Callable[[Iterable[any], int, int, Tuple[any], Mapping[str, any]], Iterable[any]],
        aggregation_method: Callable[[Iterable[Iterable[any]]], Iterable[any]] = default_aggregation,
        function_args: Tuple[any] = (),
        function_kwargs: Mapping[str, any] = {}
        ) -> Iterable[any]:
        raise NotImplementedError