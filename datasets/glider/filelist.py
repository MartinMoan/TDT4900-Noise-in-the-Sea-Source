#!/usr/bin/env python3
from cmath import inf
from fileinput import filename
import pathlib
import subprocess
import re
import sys
from datetime import datetime, timedelta

import pandas as pd
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent).working_dir)))
import config

def format_datetime(dt):
    return dt.strftime(config.DATETIME_FORMAT)

def get_wav_info(wav_filepath):
    output = subprocess.run(["ffprobe", "-i", str(wav_filepath), "-show_streams"], capture_output=True)
    if output.returncode != 0:
        print(f"Could not get information about file {wav_filepath}")
    
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
        "start_time": format_datetime(timestamp),
        "end_time": format_datetime(timestamp + timedelta(seconds=duration_seconds))
    }
    return metadata

def main(dev=False):
    data = {}
    for index, file in enumerate(config._AUDIO_FILE_LIST):
        print(index, len(config._AUDIO_FILE_LIST), file.name)
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

if __name__ == "__main__":
    main(dev = False)