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
from logger import ILogger, Logger

class IModelProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def instantiate(self) -> torch.nn.Module:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def properties(self) -> Mapping[str, any]:
        raise NotImplementedError
    
class DefaultModelProvider(IModelProvider):
    def __init__(self, model_ref: type, model_args: Tuple[any] = (), model_kwargs: Mapping[str, any] = {}, logger: ILogger = Logger(), verbose: bool = False) -> None:
        super().__init__()
        self.model_ref = model_ref
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.logger = logger
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
        sig = inspect.signature(self.model_ref)
        parameters = {}
        for index, (key, value) in enumerate(sig.parameters.items()):
            if value.default is inspect.Parameter.empty:
                # The key, value is an arg, not kwarg to the self.model_ref method
                if index >= len(self.model_args):
                    args = [key for key, v in sig.parameters.items() if v.default is inspect.Parameter.empty]
                    raise ValueError(f"The provided model args are not valid for the model class reference. Inspection of the model reference call method found {len(args)} non-keyword arguments, but only {len(self.model_args)} non-keyword arguments was provided to the {self.__class__.__name__}")
                else:
                    parameters[key] = self.model_args[index]
            else:
                # The key, value is a kwarg, not arg to the self.model_ref method
                if key in self.model_kwargs.keys():
                    parameters[key] = self.model_kwargs[key]
                else:
                    parameters[key] = value.default
        return parameters

class SomeModel(torch.nn.Module):
    def __init__(self, something, v1, v2, kw1=None, kw2: str = "???") -> None:
        super().__init__()
        
    def forward(self, X):
        return X

if __name__ == "__main__":
    provider = DefaultModelProvider(SomeModel, (1, 2, 3))
    model = provider.instantiate()
    print(provider.properties)
    