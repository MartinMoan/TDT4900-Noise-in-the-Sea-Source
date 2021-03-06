#!/usr/bin/env python3
import inspect
import hashlib
import pathlib
import pickle
import sys
import os
from typing import Type, Mapping, Tuple

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

class Cacher:
    def _args_by_inspection(obj_ref: Type, *args) -> Mapping[str, any]:
        sig = inspect.getfullargspec(obj_ref)
        sigargs = [arg for arg in sig.args if arg != "self"]
        constructor_arguments = { sigargs[i]: args[i] for i in range(min(len(sigargs), len(args))) }
        return constructor_arguments

    def _append_defaults(to_hash: Mapping[str, any]):
        uname = os.uname()
        
        hashable_arguments = {
            **to_hash,
            "ENV": str(config.ENV),
            "LOADING": "VIRTUAL_DATASET_LOADING_ENABLED" if config.VIRTUAL_DATASET_LOADING else "VIRTUAL_DATASET_LOADING_DISABLED",
            "NODENAME": str(uname.nodename)
        }
        return hashable_arguments

    def hash(self, cache_dir: pathlib.PosixPath, to_hash: Mapping[str, any]):
        hasher = hashlib.sha256()
        with_defaults = Cacher._append_defaults(to_hash)
        hasher.update(repr(with_defaults).encode())
        pickle_hash = hasher.hexdigest()
        pickle_filename = f"{str(pickle_hash)}.pickle"
        pickle_path = cache_dir.joinpath(pickle_filename).absolute()
        return pickle_path

    def cache_dir(self, stack_depth: int = 1) -> pathlib.Path:
        caller = inspect.currentframe()
        for i in range(stack_depth):
            caller = caller.f_back
        
        frameinfo = inspect.getframeinfo(caller)
        filepath = pathlib.Path(frameinfo.filename)
        
        cache_dir = config.CACHE_DIR.joinpath(filepath.stem)
        return cache_dir

    def cache(
        self,
        obj_ref: Type, 
        init_args: Tuple[any],
        init_kwargs: Mapping[str, any],
        hashable_arguments: Mapping[str, any] = None,
        force_recache: bool = False):
        
        cache_dir = self.cache_dir(stack_depth=2) # calling from withing cacher, so the second element in the stack will point to this method, but third will point to caller of this method. Therefore stack_depth = 1
        to_hash = {}
        if hashable_arguments is None:
            to_hash = {**Cacher._args_by_inspection(obj_ref, *init_args), **init_kwargs}
        else:
            to_hash = hashable_arguments
        
        pickle_path = self.hash(cache_dir, to_hash)
        cache_list = [path for path in cache_dir.glob("**/*.pickle")]
        
        if not config.CACHING_ENABLED:
            print(f"Caching is disabled")
            return self._instantiate(obj_ref, pickle_path, *init_args, **init_kwargs)
        if force_recache:
            print(f"force_recache argument flag is set, will not pickle from cache")
            return self._instantiate(obj_ref, pickle_path, *init_args, **init_kwargs)
        if not pickle_path in cache_list:
            print(f"Caching is enabled, and force_recaching is not set, but no pickle could be found in cache {cache_dir} for {obj_ref.__name__} object")
            print(f"Computed cache hash filename: {pickle_path.name}")
            return self._instantiate(obj_ref, pickle_path, *init_args, **init_kwargs)

        obj = self._load(obj_ref, pickle_path)
        return obj

    def _instantiate(self, obj_ref: Type, pickle_path: pathlib.PosixPath, *args: tuple[any], **kwargs: Mapping[str, any]) -> any:
        print(f"Instantiating new {obj_ref.__name__} object, rather than pickling from cache")
        obj = obj_ref(*args, **kwargs)
        print(f"{obj_ref.__name__} instantiated.")
        self.dump(obj, pickle_path)
        return obj

    def _load(self, obj_ref: Type, pickle_path: pathlib.PosixPath, **kwargs) -> any:
        print(f"Pickling {obj_ref.__name__} object from {pickle_path}")
        with open(pickle_path, "rb") as binary_file:
            obj = pickle.load(binary_file)
            print(f"Pickled object {obj.__class__.__name__}")
            return obj

    def dump(self, object: any, pickle_path: pathlib.PosixPath) -> None:
        print(f"Caching object {object.__class__.__name__} to {pickle_path}")
        if not pickle_path.parent.exists():
            pickle_path.parent.mkdir(parents=True, exist_ok=False)
        with open(pickle_path, "wb") as binary_file:
            pickle.dump(object, binary_file)
