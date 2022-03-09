#!/usr/bin/env python3
import subprocess
import re
from datetime import datetime, timedelta
import sys
import pathlib

import pandas as pd
import git
sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def get_wav_info(wav_filepath):
    output = subprocess.run(["ffprobe", "-i", str(wav_filepath), "-show_streams"], capture_output=True)
    if output.returncode != 0:
        print(f"Could not get information about file {wav_filepath}")
        print(output.stdout.decode("utf8"))
        print(output.stderr.decode("utf8"))
        return None
    
    decoded = output.stdout.decode("utf8")
    
    num_channels = int(re.search(r"channels=([0-9]+)\n+", decoded).group(1))
    sampling_rate = int(re.search(r"sample_rate=([0-9]+)\n+", decoded).group(1))
    num_samples = int(re.search(r"duration_ts=([0-9]+)\n+", decoded).group(1))
    duration_seconds = float(re.search(r"duration=([0-9]+\.*[0-9]*)\n+", decoded).group(1))

    filename_information = re.search(r"([^_]+)_([0-9]{3})_([0-9]{6}_[0-9]{6})\.wav", wav_filepath.name)
    
    dive_number = filename_information.group(1)
    identification_number = filename_information.group(2)

    timestring = filename_information.group(3)
    try:
        timestamp = datetime.strptime(timestring, "%y%m%d_%H%M%S")
    except Exception as ex:
        print(ex)
        timestamp = datetime(year=1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    metadata = {
        "filename": wav_filepath.relative_to(config.DATASET_DIRECTORY), 
        "num_channels": num_channels, 
        "sampling_rate": sampling_rate, 
        "num_samples": num_samples, 
        "duration_seconds": duration_seconds,
        "start_time": timestamp, # .strftime(config.DATETIME_FORMAT),
        "end_time": (timestamp + timedelta(seconds=duration_seconds))# .strftime(config.DATETIME_FORMAT)
    }
    return metadata

def get_files_df():
    filelist = config.list_local_audiofiles()
    data = [get_wav_info(filepath) for filepath in filelist]
    df = pd.DataFrame(data=data)
    return df
