#!/usr/bin/env python3
import sys
import pathlib
import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import re
from types import FunctionType

from typing import Iterable, Tuple, Mapping, Union, Callable
import math

import numpy as np
import inspect

from rich import print

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IAsyncWorker

def verify_args(decorated: FunctionType):
    def wrapper(
        self,
        iterable,
        function,
        *args,
        **kwargs):
        
        spec = inspect.getfullargspec(function)
        if len(spec.args) < 3:
            raise ValueError(f"The function argument does not seem to support (at least) 3 args (iterable, start, end). Inspection found {len(spec.args)} arguments: {spec}")
        
        if spec.annotations:
            specargs = spec.args[0:3]
            if spec.args[0] == "self":
                specargs = spec.args[1:4]
            first_arg = specargs[0] # iterable 
            second_arg = specargs[1] # start
            third_arg = specargs[2] # end
            if first_arg in spec.annotations.keys():
                annotation = spec.annotations[first_arg]
                if isinstance(annotation, type):
                    if not hasattr(annotation, "__iter__") and not (hasattr(annotation, "__getitem__") and hasattr(annotation, "__len__")):
                        raise ValueError(f"The type hint of the function argument suggests incorrect typing, must be an iterable but found {annotation}")
                else:
                    m = re.match(r"typing\.Iterable(\[.*\]){0,1}", str(annotation))
                    if m is None:
                        raise ValueError(f"The first argument to type hint of the function argument suggests incorrect typing. Expected typing.Iterable but found {annotation}")
            if second_arg in spec.annotations.keys():
                annotation = spec.annotations[second_arg]
                if annotation != int:
                    raise ValueError(f"The second argument type hint of the function argument suggest incorrect typing. Excepted second argument to be start: int but found {second_arg} : {annotation}")
            if third_arg in spec.annotations.keys():
                annotation = spec.annotations[third_arg]
                if annotation != int:
                    raise ValueError(f"The third argument type hint of the function argument suggest incorrect typing. Excepted second argument to be start: int but found {third_arg} : {annotation}")
        
        _args = (self, iterable, function, *args)
        return decorated(*_args, **kwargs)
    
    return wrapper

def progress(index: int, start: int, end: int, log_interval_percentage: float = 0.05) -> Tuple[bool, float]:
    percentage = ((index - start) / (end - start)) * 100
    part = math.ceil((end - start) * log_interval_percentage)
    return (index - start) % part == 0, percentage

class Binworker(IAsyncWorker):
    def __init__(
        self, 
        pool_ref: any = multiprocessing.pool.Pool, 
        n_processes: int = multiprocessing.cpu_count(),
        timeout_seconds: float = None) -> None:
        
        self.pool_ref = pool_ref
        self.n_processes = n_processes
        self.timeout_seconds = timeout_seconds

    def apply(
        self, 
        iterable: Iterable[any], 
        function: Callable[[Iterable[any], int, int, Tuple[any], Mapping[str, any]], Iterable[any]], 
        aggregation_method: Callable[[Iterable[Iterable[any]]], Iterable[any]] = IAsyncWorker.default_aggregation, 
        function_args: Tuple[any] = (), 
        function_kwargs: Mapping[str, any] = {}) -> Iterable[any]:
        
        """Split an iterable into (roughly) equally sized bins, and apply a function to each bin at the same time using multiple processes/threads. The results from each process/bin is aggregated and returned. 

        Args:
            iterable (Iterable[any]): The iterable to 
            function (callable): The function to apply to each bin. Must have signature (iterable: Iterable, bin_start: int, bin_end: int, *args, **kwargs) and must return an Iterable.
            aggregation_method (callable): The function to use to aggregate the bined results. Must accept a single argument (results: Iterable[Iterable[any]])
            function_args (Tuple[any], optional): Positional arguments to pass to the function (other than the required positionals). Defaults to ().
            function_kwargs (Mapping[str, any], optional): Keyword arguments to pass to the function. Defaults to {}.

        Returns:
            Iterable[any]: The aggregated result of applying the function asyncrounously over the input iterable.
        """ 
        
        with self.pool_ref(processes=self.n_processes) as pool:
            tasks = []
            binsize = math.ceil(len(iterable) / multiprocessing.cpu_count())
            for start in range(0, len(iterable), binsize):
                end = min(start + binsize, len(iterable))
                task = pool.apply_async(function, args=(iterable, start, end, *function_args), kwds=function_kwargs)
                tasks.append(task)
            
            # results = [task.get(timeout=self.timeout_seconds) for task in tasks]
            results = [task.get() for task in tasks]

            return aggregation_method(results)