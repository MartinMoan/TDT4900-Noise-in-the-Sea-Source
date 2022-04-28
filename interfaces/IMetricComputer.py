#!/usr/bin/env python3
import abc
import pathlib
import sys
from typing import Iterable, Mapping, Union

import torch
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from .ILogger import ILogger

class IMetricComputer(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def logger(self) -> ILogger:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def metrics(self) -> Iterable[callable]:
        """Return a list of """
        raise NotImplementedError

    def __call__(self, truth: torch.Tensor, preds: torch.Tensor) -> Mapping[str, Union[list, float, str]]:
        """Compute metrics given the model output (preds) and the optimal model output (truth)

        Args:
            truth (torch.Tensor): The optimal/correct model output.
            preds (torch.Tensor): The actual model output.

        Returns:
            Mapping[str, float]: a dict of {metric_name: metric_value} mappings
        """
        
        self._logger.log("Computing metrics...")
        self._logger.log(f"Truth matrix shape: {truth.shape}")
        self._logger.log(f"Prediction matrix shape: {preds.shape}")
        if len(self.metrics) == 0:
            raise Exception(f"{self.__class__.__name__} does not have any registered metric functions (self.metrics) returned iterable with length 0")

        num_failed = 0
        exceptions = []
        results = {}
        for func in self.metrics:
            result = f"{func.__name__} did not compute"
            try:
                result = func(truth.detach().numpy(), preds.detach().numpy())
            except Exception as ex: 
                result = f"{func.__name__} failed to compute with the following exception: {str(ex)}"
                exceptions.append(ex)
                num_failed += 1

            results[func.__name__] = result
            
        if num_failed == len(self.metrics) and len(self.metrics) != 0:
            caught_exceptions = "\n".join([str(ex) for ex in exceptions])
            raise Exception(f"{self.__class__.__name__}: No metrics could be computed.\n{caught_exceptions}")
        self._logger.log("Metric computation done!")
        self._logger.log(results)
        return results