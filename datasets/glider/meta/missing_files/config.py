#!/usr/bin/env python3
import sys
import pathlib
from dataclasses import dataclass
import os
import warnings

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import CustomWarnings
warnings.formatwarning = CustomWarnings.custom_format

@dataclass
class Config:
    _DATASET_SAS=os.environ.get("GLIDER_SharedAccessSignature")
    _DATASET_REMOTE_URL=os.environ.get("GLIDER_BLOB_URL")
    _DATASET_CONTAINER_NAME=os.environ.get("GLIDER_CONTAINER_NAME")
    _DATASET_REMOTE_PREFIX=os.environ.get("GLIDER_BLOB_PATH") #used 
    _DATASET_MAX_FILE_DOWNLOAD_ATEMPTS=3

    if _DATASET_SAS is None:
        warnings.warn("Missing environment variable 'GLIDER_SharedAccessSignature' from .env. This can cause unexpected errors when trying to communicate with the remote dataset blobstorage.")
    if _DATASET_REMOTE_URL is None:
        warnings.warn("Missing environment variable 'GLIDER_BLOB_URL' from .env. This can cause unexpected errors when trying to communicate with the remote dataset blobstorage.")
    if _DATASET_CONTAINER_NAME is None:
        warnings.warn("Missing environment variable 'GLIDER_CONTAINER_NAME' from .env. This can cause unexpected errors when trying to communicate with the remote dataset blobstorage.")
    if _DATASET_REMOTE_PREFIX is None:
        warnings.warn("Missing environment variable 'GLIDER_BLOB_PATH' from .env. This can cause unexpected errors when trying to communicate with the remote dataset blobstorage.")

    DATASET_URL=f"{_DATASET_REMOTE_URL}/{_DATASET_CONTAINER_NAME}?{_DATASET_SAS}"