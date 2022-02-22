#!/usr/bin/env python3
import pathlib
import sys
import subprocess
import re
import shutil
from datetime import datetime

import pandas as pd
import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def download_missing_files(filelimit=None):
    cmd = ["azcopy", "list", config.DATASET_URL, "--properties", "ContentType", "--machine-readable"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    content = process.stdout

    lines = re.split(r"\n+", content)
    
    data = {}
    for line in lines:
        matches = re.findall(r"([^:]+):\s*([^;]+);*\s*", line)
        for match in matches:
            key, value = match
            if key not in data.keys():
                data[key] = [value]
            else:
                data[key].append(value)

    df = pd.DataFrame(data=data)
    df["Content Length"] = df["Content Length"].astype(int)
    df = df.sort_values(by="ContentType")
    
    audio = df[(df["ContentType"] == "audio/x-wav")].copy()
    not_audio = df[(df["ContentType"] != "audio/x-wav")]
    total_size_bytes = audio["Content Length"].sum()

    audio["remotePath"] = audio["INFO"]
    filenames = [pathlib.Path(remotepath).name for remotepath in audio["remotePath"].values]
    audio["filename"] = filenames

    local_files = pd.DataFrame(data={"filename": [pathlib.Path(filepath).name for filepath in config.list_local_audiofiles()]})

    missing_files = audio[(~audio["filename"].isin(local_files["filename"]))]
    missing_bytes = missing_files["Content Length"].sum()

    missing_files.to_csv(config._META_MISSING_FILES_DIRECTORY.joinpath("remaining_files.csv"), index=False)

    total_gb = total_size_bytes / (1000**3)
    missing_gb = missing_bytes / (1000**3)
    local_gb = (total_size_bytes - missing_bytes) / (1000**3)

    print(f"Total remote dataset size {len(audio)} audio files totaling {total_gb} GB")
    print(f"Total missing data {len(missing_files)} files totaling {missing_gb} GB")

    missing_files_list = missing_files["INFO"].values
    if filelimit is None:
        filelimit = len(missing_files_list)
    to_download = "\n".join(missing_files_list[:filelimit])
    
    missing_files_txt_path = config._META_MISSING_FILES_DIRECTORY.joinpath("missing.txt")
    file = open(missing_files_txt_path, "w")
    file.write(to_download)
    file.close()
    
    cmd = ["azcopy", "copy", str(config.DATASET_URL), str(config.TMP_DATA_DIR), "--list-of-files", str(missing_files_txt_path)]
    print(" ".join(cmd))
    subprocess.run(cmd)

def cleanup_tempfiles():
    tmp_files = list(config.TMP_DATA_DIR.glob("**/*.wav"))
    for file in tmp_files:
        filename_information = re.search(r"([^_]+)_([0-9]{3})_([0-9]{6}_[0-9]{6})\.wav", file.name)
        dive_number = filename_information.group(1)
        identification_number = filename_information.group(2)

        timestring = filename_information.group(3)
        try:
            timestamp = datetime.strptime(timestring, "%y%m%d_%H%M%S")
        except Exception as ex:
            print(ex)
            timestamp = datetime(year=1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        month_name = timestamp.strftime("%B")
        destination = config._GLIDER_DATASET_DIRECTORY.joinpath(month_name, file.name)
        if not destination.parent.exists():
            destination.parent.mkdir(parents=False, exist_ok=True)
        shutil.move(file, destination)

def main():
    # download_missing_files()
    cleanup_tempfiles()

if __name__ == "__main__":
    main()