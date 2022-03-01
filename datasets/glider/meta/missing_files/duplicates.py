#!/usr/bin/env python3
#!/usr/bin/env python3
import multiprocessing
from os import dup
import pathlib
import sys
import subprocess
from multiprocessing.pool import ThreadPool, Pool
import re
import shutil
from datetime import datetime

import pandas as pd
import numpy as np
import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def main():
    files = config.list_local_audiofiles()
    filenames = [file.name for file in files]
    unique_filenames, counts = np.unique(filenames, return_counts=True)
    duplicates = unique_filenames[counts > 1]
    print(f"Out of {len(files)} local audio files, there are {len(duplicates)} duplicate files")
    for file in duplicates:
        print(file)

if __name__ == "__main__":
    print(__file__)
    main()