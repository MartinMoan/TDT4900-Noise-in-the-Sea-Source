#!/usr/bin/env python3
import sys 
import pathlib
from typing import Iterable

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import IFileListProvider

class AudioFileListProvider(IFileListProvider):
    def __init__(self) -> None:
        super().__init__()

    def list(self) -> Iterable[pathlib.Path]:
        # return list(config.TMP_DATA_DIR.glob("**/*.wav")) + list(config.DATASET_DIRECTORY.glob("**/*.wav"))
        return list(config.DATASET_DIRECTORY.glob("**/*.wav"))

if __name__ == "__main__":
    filelistprovider = AudioFileListProvider()
    files = filelistprovider.list()
    print(files)
    
# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
# print(config.DATASET_URL)
# client = ContainerClient.from_container_url(config.DATASET_URL)
# blobs = list(client.list_blobs(config._DATASET_REMOTE_PREFIX))
# required_content_type="audio/x-wav"

# audioblobs = []
# for blob in blobs:
#     content_settings = blob["content_settings"]
#     if content_settings is not None:
#         content_type=content_settings["content_type"]
#         if content_type is not None and content_type == required_content_type:
#             audioblobs.append(blob)

# print(f"Found {len(blobs)} blobs in blobstorage, including non-audio files.")
# print(f"Found {len(audioblobs)} blobs with content type {required_content_type}")
# missing_blobs = []
# for blob in audioblobs:
#     remote_relative_filepath = pathlib.Path(blob["name"])
#     print(remote_relative_filepath)
#     filename = remote_relative_filepath.name
#     if filename not in local_files:
#         missing_blobs.append(blob)
# # print(f"Of these there are {len(missing_blobs)} missing from the local dataset directory")
# return missing_blobs