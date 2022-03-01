#!/usr/bin/env python3
import multiprocessing
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
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

def get_container_client():
    return ContainerClient.from_container_url(config.DATASET_URL)

def get_audioblobs(bloblist, required_content_type="audio/x-wav"):
    audio_blobs = []
    for blob in bloblist:
        content_settings = blob["content_settings"]
        if content_settings is not None:
            content_type=content_settings["content_type"]
            if content_type is not None and content_type == required_content_type:
                audio_blobs.append(blob)
    return audio_blobs

def missing_remote_blobs(client, local_filenames):
    blobs = list(client.list_blobs(config._DATASET_REMOTE_PREFIX))
    required_content_type="audio/x-wav"
    audioblobs = get_audioblobs(blobs, required_content_type="audio/x-wav")
    print(f"Found {len(blobs)} blobs in blobstorage, including non-audio files.")
    print(f"Found {len(audioblobs)} blobs with content type {required_content_type}")
    missing_blobs = []
    for blob in audioblobs:
        remote_relative_filepath = pathlib.Path(blob["name"])
        filename = remote_relative_filepath.name
        if filename not in local_filenames:
            missing_blobs.append(blob)
    # print(f"Of these there are {len(missing_blobs)} missing from the local dataset directory")
    return missing_blobs

class MultiThreadAzureBlobDownloader:
    def _verified(self, local_filepath):
        output = subprocess.run(["ffprobe", "-i", str(local_filepath), "-show_streams"], capture_output=True)
        return output.returncode == 0

    def _download_atempt(self, pid, blob, local_tmp_filepath):
        client = get_container_client()
        downloader = client.download_blob(blob, offset=0, length=blob.size, timeout=60)
        with open(local_tmp_filepath, "wb") as file:
            props = downloader.download_to_stream(file)
            print(f"Process {pid} done! File downloaded to {local_tmp_filepath.absolute()}")

    def _download(self, blob):
        filename = pathlib.Path(blob.name).name
        local_tmp_filepath = config.TMP_DATA_DIR.joinpath(filename)
        process = multiprocessing.current_process()
        print(f"Process {process.pid} downloading remote file {filename} to {local_tmp_filepath.absolute()}")

        try:
            for atempt in range(config._DATASET_MAX_FILE_DOWNLOAD_ATEMPTS):
                self._download_atempt(process.pid, blob, local_tmp_filepath)
                print(f"Process {process.pid} - verifying local file integrity {local_tmp_filepath.absolute()}")
                if not self._verified(local_tmp_filepath):
                    print(f"Process {process.pid} - could not verify file integrity, retrying...")
                else: 
                    return True, local_tmp_filepath
            raise Exception(f"Failed to download and verify file {filename} from blob after {config._DATASET_MAX_FILE_DOWNLOAD_ATEMPTS} retries")
        except Exception as ex:
            print(f"Process {process.pid} failed downloading remote file {filename} to {local_tmp_filepath.absolute()}")
            print(str(ex))
            return False, local_tmp_filepath 

    def download(self, blobs):
        total_size = np.sum([blob.size for blob in blobs])
        print(f"Beginning download of {len(blobs)} Azure storage blobs missing from local directory.")
        print(f"With a total size of {bytes_to_gb(total_size):.2f} GB")
        print()
        with Pool(processes=int(multiprocessing.cpu_count() - 1)) as pool:
            return pool.map(self._download, blobs)

def get_local_audiofiles():
    return pd.DataFrame(data={"filename": [pathlib.Path(filepath).name for filepath in config.list_local_audiofiles()]})

def bytes_to_gb(num_bytes):
    return num_bytes / (1000**3)

def download_missing_files():
    local_files = get_local_audiofiles()
    local_filenames = local_files["filename"].values
    
    client = get_container_client()
    missing_blobs = missing_remote_blobs(client, local_filenames)
    downloader = MultiThreadAzureBlobDownloader()
    print(f"There are {len(local_filenames)} local audiofiles.")
    print(f"With {len(missing_blobs)} files present in blobstorage not present amongst local files")
    downloaded_files = downloader.download(missing_blobs)
    return downloaded_files

# def cleanup_tempfiles():
#     tmp_files = list(config.TMP_DATA_DIR.glob("**/*.wav"))
#     for file in tmp_files:
#         filename_information = re.search(r"([^_]+)_([0-9]{3})_([0-9]{6}_[0-9]{6})\.wav", file.name)
#         dive_number = filename_information.group(1)
#         identification_number = filename_information.group(2)

#         timestring = filename_information.group(3)
#         try:
#             timestamp = datetime.strptime(timestring, "%y%m%d_%H%M%S")
#         except Exception as ex:
#             print(ex)
#             continue

#         month_name = timestamp.strftime("%B")
#         destination = config._GLIDER_DATASET_DIRECTORY.joinpath(month_name, file.name)
#         if not destination.parent.exists():
#             destination.parent.mkdir(parents=False, exist_ok=True)
#         shutil.move(file, destination)

def main():
    missing_files = download_missing_files()
    print(missing_files)

if __name__ == "__main__":
    print(__file__)
    main()