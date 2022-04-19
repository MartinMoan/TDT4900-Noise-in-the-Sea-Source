#!/usr/bin/env python3
import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import re
from types import FunctionType

from typing import Iterable, Tuple, Mapping, Union
import math

import numpy as np
import inspect

from rich import print

from logger import Logger

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
            first_arg = spec.args[0] # iterable 
            second_arg = spec.args[1] # start
            third_arg = spec.args[2] # end
            if first_arg in spec.annotations.keys():
                annotation = str(spec.annotations[first_arg])
                m = re.match(r"typing\.Iterable(\[.*\]){0,1}", annotation)
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
        decorated(*_args, **kwargs)
    
    return wrapper

class Binworker:
    def __init__(
        self, 
        pool_ref: any = multiprocessing.pool.Pool, 
        n_processes: int = multiprocessing.cpu_count()) -> None:
        
        self.pool_ref = pool_ref
        self.n_processes = n_processes

    def default_aggregation(results: Iterable[Iterable[any]]) -> Iterable[any]:
        output = []
        for iterable in results:
            for result in iterable:
                output.append(result)
        return output

    @verify_args
    def apply(
        self, 
        iterable: Iterable[any], 
        function: callable, 
        function_args: Tuple[any] = (), 
        function_kwargs: Mapping[str, any] = {},
        aggregation_method: callable = default_aggregation) -> Iterable[any]:
        """Split an iterable into (roughly) equally sized bins, and apply a function to each bin at the same time using multiple processes/threads. The results from each process/bin is aggregated and returned. 

        Args:
            iterable (Iterable[any]): The iterable to 
            function (callable): The function to apply to each bin. Must have signature (iterable: Iterable, bin_start: int, bin_end: int, *args, **kwargs) and must return an Iterable.
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
            
            results = [task.get() for task in tasks]

            return aggregation_method(results)

def square(iterable: Iterable[Union[float, int]], start: int, end: int, *args, **kwargs) -> Iterable[Union[float, int]]:
    logger = Logger()
    logger.log(f"start {start} end {end}")
    output = []
    for i in range(start, end):
        output.append(iterable[i]**2)
    return output

def noannotations(iterable, start, end):
    return square(iterable, start, end)

if __name__ == "__main__":
    worker = Binworker(pool_ref=multiprocessing.Pool)
    worker.apply(np.arange(100), square)