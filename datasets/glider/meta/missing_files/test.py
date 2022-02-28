#!/usr/bin/env python3
import pathlib
import sys
import subprocess
import re
import shutil
from datetime import datetime
from matplotlib import container

import pandas as pd
import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__, generate_account_sas

print(config._DATASET_REMOTE_PATH, config._DATASET_SAS, config.DATASET_URL)
client = BlobServiceClient(
    account_url=config.DATASET_URL
)

print(client)
for container in client.list_containers():
    print(container)
# container_client = ContainerClient.from_container_url(config._DATASET_REMOTE_PATH, credential=config._DATASET_SAS)

# for key in dir(container_client):
#     print(key, getattr(container_client, key))