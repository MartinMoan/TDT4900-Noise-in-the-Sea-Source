#!/usr/bin/env python3
import multiprocessing
import pathlib
import subprocess
from multiprocessing.pool import Pool, ThreadPool
import re
import sys
from datetime import datetime, timedelta
from utils import get_wav_info
import pandas as pd
import numpy as np
import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def _info(args):
    index, max, filepath = args
    if index % 100 == 0:
        print(index, max, filepath)
    return get_wav_info(filepath)
        
def get_file_info(local_filelist):
    with Pool(processes=int(multiprocessing.cpu_count() - 1)) as pool:
        ids = range(len(local_filelist))
        maxid = np.full(len(local_filelist), len(local_filelist))
        items = zip(ids, maxid, local_filelist)
        return pool.map(_info, items)

def get_filelist(dev=False, reload_files=True):
    if reload_files:
        data = {}
        local_filelist = config.list_local_audiofiles()
        fileinfo = get_file_info(local_filelist)
        print(fileinfo)
        exit()
        for index, info in enumerate(fileinfo):            
            if info is not None:
                for key in info.keys():
                    if key not in data.keys():
                        data[key] = [info[key]]
                    else:
                        data[key].append(info[key])

                if dev and index >= 100:
                    break

        df = pd.DataFrame(data=data)
        df.to_csv(config.AUDIO_FILE_CSV_PATH, index=False)
        print(df)
        return df
    else:
        df = pd.read_csv(config.AUDIO_FILE_CSV_PATH)
        return df

def main():
    df = get_filelist(reload_files=True)

if __name__ == "__main__":
    main()
    