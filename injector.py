#!/usr/bin/env python3

class Injector:
    def __init__(self, *args, **kwargs) -> None:
        print("Injector.__init__", args, kwargs)

    def __call__(self, *args, **kwargs):
        print("Injector.__call__", args, kwargs)