#!/usr/bin/env python3
from fileinput import filename
import pathlib
import subprocess
import re
import sys
from datetime import datetime

import pandas as pd

config_path = pathlib.Path(__file__).parent.joinpath("config.py").absolute()
sys.path.insert(0, str(config_path))
import config

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
    
    idk_what_this_is = filename_information.group(1)
    idk_what_this_is_2 = filename_information.group(2)
    timestring = filename_information.group(3)
    timestamp = datetime.strptime(timestring, "%y%m%d_%H%M%S") # "%H%M%S_%d%m%y"
    
    metadata = {
        "filename": wav_filepath.relative_to(config.DATASET_DIRECTORY), 
        "num_channels": num_channels, 
        "sampling_rate": sampling_rate, 
        "num_samples": num_samples, 
        "duration_seconds": duration_seconds,
        "timestamp": timestamp,
    }
    return metadata

def main():
    data = {}
    for index, file in enumerate(config._AUDIO_FILE_LIST):
        print(index, len(config._AUDIO_FILE_LIST))
        info = get_wav_info(file)

        for key in info.keys():
            if key not in data.keys():
                data[key] = [info[key]]
            else:
                data[key].append(info[key])


        if index >= 100:
            break

    df = pd.DataFrame(data=data)
    df.to_csv(config.AUDIO_FILE_CSV_PATH, index=False)

if __name__ == "__main__":
    main()