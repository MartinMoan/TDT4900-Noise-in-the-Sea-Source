#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import pathlib
import sys
import re
import multiprocessing
from typing import Iterable

from rich import print
import pandas as pd
import numpy as np
import git
import librosa

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from datasets.glider.wavfileinspector import WaveFileInspector
from tqdm import tqdm

def check(data):
    index, filepath = data
    print(index)
    info = WaveFileInspector.info(filepath)
    samples, sr = librosa.load(filepath, sr=None)
    dur_sec = len(samples) / sr
    if dur_sec != info.duration_seconds:
        raise Exception(f"{filepath} is corrupted. The number of samples stated in the file header does not match the number of samples found in the file contents")

def main(args):
    files = config.list_local_audiofiles()
    print(len(files))

    with multiprocessing.Pool(processes=args.n_processors) as pool:
        pool.map(check, zip(range(len(files)), files))

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_processors", type=int, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = init() 
    main(args)