#!/usr/bin/env python3
import inspect
import abc
import sys
import pathlib
from turtle import forward
from typing import Tuple, Mapping, Iterable

import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import IModelProvider, ILoggerFactory
    
class DefaultModelProvider(IModelProvider):
    def __init__(self, model_ref: type, model_args: Tuple[any] = (), model_kwargs: Mapping[str, any] = {}, logger_factory: ILoggerFactory = None, verbose: bool = False) -> None:
        super().__init__()
        self.model_ref = model_ref
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.logger = logger_factory.create_logger()
        self.verbose = verbose

        sig = inspect.signature(self.model_ref)
        args = [key for key, v in sig.parameters.items() if v.default is inspect.Parameter.empty]
        if len(args) != len(self.model_args):
            raise ValueError(f"The provided model args are not valid for the model class reference. Inspection of the model reference call method found {len(args)} non-keyword arguments, but only {len(self.model_args)} non-keyword arguments was provided to the {self.__class__.__name__}")

    def instantiate(self) -> torch.nn.Module:
        if self.verbose:
            self.logger.log("Instantiating model...")
            model = self.model_ref(*self.model_args, **self.model_kwargs)
            self.logger.log("Model instantiated!")
            return model
        return self.model_ref(*self.model_args, **self.model_kwargs)

    @property
    def properties(self) -> Mapping[str, any]:
        parameters = {
            "model_ref": self.model_ref,
            "model_args": self.model_args,
            "model_kwargs": self.model_kwargs,
        }
        return parameters

class SomeModel(torch.nn.Module):
    def __init__(self, something, v1, v2, kw1=None, kw2: str = "???") -> None:
        super().__init__()
        
    def forward(self, X):
        return X

if __name__ == "__main__":
    from tracking.loggerfactory import LoggerFactory
    from tracking.logger import Logger

    factory = LoggerFactory(logger_type=Logger)
    logger = factory.create_logger()
    provider = DefaultModelProvider(SomeModel, model_args=(1, 2, 3), logger_factory=factory)
    model = provider.instantiate()
    logger.log(provider.properties)
    