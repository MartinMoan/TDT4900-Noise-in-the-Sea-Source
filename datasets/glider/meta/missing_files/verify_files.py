#!/usr/bin/env python3
import multiprocessing
import pathlib
import sys
import subprocess
from multiprocessing.pool import ThreadPool, Pool
import re
import shutil
from datetime import datetime
import os
from unittest import result

import pandas as pd
import numpy as np
import git
from rich import print
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from utils import get_wav_info

def _verify(filepath):
    output = subprocess.run(["ffprobe", "-i", str(filepath), "-show_streams"], capture_output=True)
    return {"verified": output.returncode == 0, "path": filepath}

def verify(files):
    with Pool(processes=int(multiprocessing.cpu_count() - 1)) as pool:
        return pool.map(_verify, files)

def get_tmp_files():
    print(f"Looking for .wav files in tmp data directory {str(config.TMP_DATA_DIR)}...")
    tmp_files = list(config.TMP_DATA_DIR.glob("**/*.wav"))
    print(f"Found {len(tmp_files)} .wav files in tmp directory")
    return tmp_files

def get_verified_tmp_files(verification_results):
    return [result["path"] for result in verification_results if result["verified"]]

def cleanup_tmp_files():
    tmp_files = get_tmp_files()
    results = verify(tmp_files)
    
    verified_files = get_verified_tmp_files(results)

    for verified in verified_files:
        info = get_wav_info(verified)
        month = info["start_time"].strftime("%B")
        data_dir = config._GLIDER_DATASET_DIRECTORY.joinpath(month)

        if not data_dir.exists() and not data_dir.is_file():
            data_dir.mkdir(parents=False, exist_ok=False)

        filepath = data_dir.joinpath(info["filename"].name)
        src, dest = info["filename"], filepath
        print(f"Moving verified file {src.name} from {src} to {dest}")
        shutil.move(src, dest)

def verify_non_tmp_files():
    audiofiles = list(config.DATASET_DIRECTORY.glob("**/*.wav"))
    results = verify(audiofiles)
    not_verified = [result["path"] for result in results if not result["verified"]]
    for file in not_verified:
        print(f"File {str(file)} could not be verified, removing...")
        os.remove(file)

def main():
    cleanup_tmp_files()
    verify_non_tmp_files()

if __name__ == "__main__":
    main()