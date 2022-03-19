#!/usr/bin/env python3
import pathlib
import sys
from time import perf_counter

from rich import print
import git
sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from GLIDER import GLIDER

def time(func, *args, **kwargs):
    start = perf_counter()
    output = func(*args, **kwargs)
    end = perf_counter()
    duration = end - start
    return duration, output

def main():
    _, glider = time(GLIDER)
    total = 0
    for i in range(len(glider)):
        duration, data = time(glider.__getitem__, i)
        if i % 100 == 0:
            print(f"{i} / {len(glider)} - Total time spent: {total} seconds. Average per it. {total / (i+1)}")
        total += duration
    



if __name__ == "__main__":
    main()