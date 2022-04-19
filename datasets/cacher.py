#!/usr/bin/env python3
from functools import cache
import hashlib
import pathlib
import pickle
import sys
import os
from typing import Type, Mapping

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from logger import Logger, ILogger

class Cacher:
    def __init__(self, logger: ILogger = Logger()) -> None:
        self.logger = logger
        
    def _get_hashable_arguments(*args, **kwargs):
        input_arguments = tuple([arg for arg in args] + [kwargs[key] for key in kwargs.keys()])
        uname = os.uname()
        
        hashable_arguments = tuple((
            str(config.ENV),
            "VIRTUAL_DATASET_LOADING_ENABLED" if config.VIRTUAL_DATASET_LOADING else "VIRTUAL_DATASET_LOADING_DISABLED",
            str(uname.nodename),
            *input_arguments
        ))
        return hashable_arguments

    def cache(
        self,
        cache_dir: pathlib.PosixPath, 
        obj_ref: Type, 
        *args: tuple[any], 
        hashable_arguments: tuple[any] = None, 
        force_recache: bool = False,
        **kwargs: Mapping[str, any]):
        
        hasher = hashlib.sha256()
        if hashable_arguments is None:
            hasher.update(repr(Cacher._get_hashable_arguments(*args, **kwargs)).encode())
        else:
            hasher.update(hashable_arguments)
        pickle_hash = hasher.hexdigest()
        pickle_filename = f"{str(pickle_hash)}.pickle"

        cache_list = [path for path in cache_dir.glob("**/*.pickle")]
        pickle_path = cache_dir.joinpath(pickle_filename).absolute()
        
        if config.CACHING_ENABLED and not force_recache and pickle_path in cache_list:
            self.logger.log(f"Pickling {obj_ref.__name__} object from {pickle_path}")
            with open(pickle_path, "rb") as binary_file:
                obj = pickle.load(binary_file)
                return obj
        else:
            self.logger.log(f"Instantiating new {obj_ref.__name__} object, rather than pickling from cache")
            obj = obj_ref(*args, **kwargs)
            with open(pickle_path, "wb") as binary_file:
                pickle.dump(obj, binary_file)
            return obj