#!/usr/bin/env python3
import pathlib
import subprocess
import re
import sys
from datetime import datetime, timedelta
from utils import get_wav_info
import pandas as pd
import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def get_filelist(dev=False, reload_files=True):
    if reload_files:
        data = {}
        local_filelist = config.list_local_audiofiles()
        for index, file in enumerate(local_filelist):
            print(index, len(local_filelist), file.name)
            info = get_wav_info(file)

            for key in info.keys():
                if key not in data.keys():
                    data[key] = [info[key]]
                else:
                    data[key].append(info[key])

            if dev and index >= 100:
                break

        df = pd.DataFrame(data=data)
        df.to_csv(config.AUDIO_FILE_CSV_PATH, index=False)
        return df
    else:
        df = pd.read_csv(config.AUDIO_FILE_CSV_PATH)
        return df

def main():
    df = get_filelist(reload_files=True)

if __name__ == "__main__":
    main()
    