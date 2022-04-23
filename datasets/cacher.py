#!/usr/bin/env python3
from functools import cache
import hashlib
import pathlib
import pickle
import sys
import os
from typing import Type, Mapping

import git

from dependency_injector.wiring import Provide, inject

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from logger import Logger, ILogger

from globalcontainer import GlobalContainer
from globalconfig import GlobalConfiguration

class Cacher:
    def __init__(self, logger: ILogger = Provide[GlobalContainer.logger], config: GlobalConfiguration = Provide[GlobalContainer.config]) -> None:
        self.logger = logger
        self.config = config
        
    def _get_hashable_arguments(self, *args, **kwargs):
        input_arguments = tuple([arg for arg in args] + [kwargs[key] for key in kwargs.keys()])
        uname = os.uname()
        
        hashable_arguments = tuple((
            str(self.config.ENV),
            "VIRTUAL_DATASET_LOADING_ENABLED" if self.config.VIRTUAL_DATASET_LOADING else "VIRTUAL_DATASET_LOADING_DISABLED",
            str(uname.nodename),
            *input_arguments
        ))
        return hashable_arguments

    def hash(self, cache_dir: pathlib.PosixPath, *args, hashable_arguments: tuple[any] = None, **kwargs):
        hasher = hashlib.sha256()
        if hashable_arguments is None:
            hasher.update(repr(self._get_hashable_arguments(*args, **kwargs)).encode())
        else:
            hasher.update(hashable_arguments)
        pickle_hash = hasher.hexdigest()
        pickle_filename = f"{str(pickle_hash)}.pickle"
        pickle_path = cache_dir.joinpath(pickle_filename).absolute()
        return pickle_path

    def cache(
        self,
        cache_dir: pathlib.PosixPath, 
        obj_ref: Type, 
        *args: tuple[any], 
        hashable_arguments: tuple[any] = None, 
        force_recache: bool = False,
        **kwargs: Mapping[str, any]):
        
        pickle_path = self.hash(cache_dir, *args, hashable_arguments=hashable_arguments, **kwargs)
        cache_list = [path for path in cache_dir.glob("**/*.pickle")]
        
        if not self.config.CACHING_ENABLED:
            self.logger.log(f"Caching is disabled")
            return self._instantiate(obj_ref, pickle_path, *args, **kwargs)
        if force_recache:
            self.logger.log(f"force_recache argument flag is set, will not pickle from cahche")
            return self._instantiate(obj_ref, pickle_path, *args, **kwargs)
        if not pickle_path in cache_list:
            self.logger.log(f"Caching is enabled, and force_recahcing is not set, but no pickle could be found in cache for {obj_ref.__name__} object")
            self.logger.log(f"Computed cache hash filename: {pickle_path.name}")
            return self._instantiate(obj_ref, pickle_path, *args, **kwargs)

        return self._load(obj_ref, pickle_path)

    def _instantiate(self, obj_ref: Type, pickle_path: pathlib.PosixPath, *args: tuple[any], **kwargs: Mapping[str, any]) -> any:
        self.logger.log(f"Instantiating new {obj_ref.__name__} object, rather than pickling from cache")
        obj = obj_ref(*args, **kwargs)
        self.logger.log(f"{obj_ref.__name__} instantiated, caching object to {pickle_path}")
        with open(pickle_path, "wb") as binary_file:
            pickle.dump(obj, binary_file)
        return obj

    def _load(self, obj_ref: Type, pickle_path: pathlib.PosixPath) -> any:
        self.logger.log(f"Pickling {obj_ref.__name__} object from {pickle_path}")        
        with open(pickle_path, "rb") as binary_file:
            obj = pickle.load(binary_file)
            self.logger.log(f"Pickled object {obj}")
            return obj

    def dump(self, object: any, pickle_path: pathlib.PosixPath) -> None:
        self.logger.log(f"Dumping object {object} to {pickle_path}")
        with open(pickle_path, "wb") as binary_file:
            pickle.dump(object, binary_file)
