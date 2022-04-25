#!/usr/bin/env python3
import sys
import pathlib
import inspect

import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

def verify(func):
    print("func", func)
    def nested(*args, **kwargs):
        print(args, kwargs)
        print(dir(args[0]))
        spec = inspect.getfullargspec(func)
        names = spec.args
        named_args = {names[i]: args[i] for i in range(len(names))}
        named_args = {key: value for key, value in named_args.items() if key != "self"}
        named_args = {**named_args, **kwargs}
        
        for arg, input_value in named_args.items():
            if arg in spec.annotations:
                expected_type = spec.annotations[arg]
                if not isinstance(input_value, expected_type):
                    raise TypeError(f"Argument {arg} has incorrect type. Expected {expected_type} but received {type(input_value)}")

        func(*args, **kwargs)
    return nested

