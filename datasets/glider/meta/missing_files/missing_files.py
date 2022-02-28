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
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

def azcopy_installed():
    return shutil.which("azcopy") is not None

def list_remote_files():
    cmd = ["azcopy", "list", f'{config.DATASET_URL}', "--machine-readable"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        command = " ".join(cmd)
        raise Exception(f"Could not list files in storage blob. Command {command} failed with non-zero returncode.\n{process.stdout}")
    
    content = process.stdout    
    lines = re.split(r"\n+", content)
    
    min_matches = 3 # 3 required key/value pairs from azcopy list command (INFO, ContentType, 'Content Length')
    data = {}
    for line in lines:
        matches = re.findall(r"([^:]+):\s*([^;]+);*\s*", line)
        if len(matches) == min_matches:
            for match in matches:
                key, value = match
                if key not in data.keys():
                    data[key] = [value]
                else:
                    data[key].append(value)
    
    df = pd.DataFrame(data=data)
    df["Content Length"] = df["Content Length"].astype(int)
    df["ContentType"] = [pathlib.Path(path).suffix for path in df["INFO"].values]
    df = df.sort_values(by="ContentType")
    return df

def get_remote_audiofiles(missing_remote_files_df):
    df = missing_remote_files_df
    audio = df[(df["ContentType"] == ".wav")].copy()
    not_audio = df[(df["ContentType"] != ".wav")]

    audio["remotePath"] = audio["INFO"]
    filenames = [pathlib.Path(remotepath).name for remotepath in audio["remotePath"].values]
    audio["filename"] = filenames
    return audio

def get_local_audiofiles():
    return pd.DataFrame(data={"filename": [pathlib.Path(filepath).name for filepath in config.list_local_audiofiles()]})

def bytes_to_gb(num_bytes):
    return num_bytes / (1000**3)

def get_missing_files():
    df = list_remote_files()
    audio = get_remote_audiofiles(df)
    
    local_files = get_local_audiofiles()

    missing_files = audio[(~audio["filename"].isin(local_files["filename"]))]

    missing_files.to_csv(config._META_MISSING_FILES_DIRECTORY.joinpath("remaining_files.csv"), index=False)
    logmessage(audio, missing_files)
    return missing_files

def logmessage(audio, missing_files):
    total_size_bytes = audio["Content Length"].sum()
    missing_bytes = missing_files["Content Length"].sum()

    total_gb = bytes_to_gb(total_size_bytes)
    missing_gb = bytes_to_gb(missing_bytes)

    print(f"Total remote dataset size {len(audio)} audio files totaling {total_gb} GB")
    print(f"Total missing data {len(missing_files)} files totaling {missing_gb} GB")

def download_missing_files(missing_files):
    missing_files_list = missing_files["INFO"].values
    print(f"Begginning to download {len(missing_files)} audiofiles to directory {str(config.TMP_DATA_DIR)}")
    to_download = "\n".join(missing_files_list)
    missing_files_txt_path = config._META_MISSING_FILES_DIRECTORY.joinpath("missing.txt")
    file = open(missing_files_txt_path, "w")
    file.write(to_download)
    file.close()
    cmd = ["azcopy", "copy", str(config.DATASET_URL), str(config.TMP_DATA_DIR), "--list-of-files", str(missing_files_txt_path)]
    command = " ".join(cmd)
    print(command)
    process = subprocess.run(cmd, stdin=to_download)

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
    if not azcopy_installed():
        raise Exception("azcopy is not installed. Install and add it to the system path and retry.")
    
    missing_files = get_missing_files()
    print(missing_files)
    download_missing_files(missing_files)
    cleanup_tempfiles()

if __name__ == "__main__":
    main()