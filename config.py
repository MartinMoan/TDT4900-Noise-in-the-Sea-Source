#!/usr/bin/env python3
from inspect import trace
from logging import warning
import pathlib
import getpass
import sys
import os
from types import TracebackType
import warnings
from urllib.parse import unquote

import dotenv
dotenv.load_dotenv()

import git
from rich import print

import CustomWarnings
ENV = os.environ.get("ENV", "dev")
if ENV == "dev":
    # warnings.warn = CustomWarnings.warn
    warnings.formatwarning = CustomWarnings.custom_format

repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
REPO_DIR = pathlib.Path(repo.working_dir).absolute()

_DATASET_SAS=os.environ.get("GLIDER_SharedAccessSignature")
_DATASET_REMOTE_URL=os.environ.get("GLIDER_BLOB_URL")
_DATASET_CONTAINER_NAME=os.environ.get("GLIDER_CONTAINER_NAME")
_DATASET_REMOTE_PREFIX=os.environ.get("GLIDER_BLOB_PATH")
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

for directory in [d for d in REPO_DIR.glob("**/*") if d.is_dir()]:
    sys.path.insert(0, str(directory))

_GLIDER_DATASET_LABELS_DIRECTORY = REPO_DIR.joinpath("datasets", "glider")

_PARSED_LABELS_CSV_FILENAME = "glider_labels.csv"
_PARSED_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("parsed")
_PARSED_DIRECTORY.mkdir(parents=False, exist_ok=True)
_META_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("meta")
_META_DIRECTORY.mkdir(parents=False, exist_ok=True)
_META_MISSING_FILES_DIRECTORY = _META_DIRECTORY.joinpath("missing_files")
_META_MISSING_FILES_DIRECTORY.mkdir(parents=False, exist_ok=True)
PARSED_LABELS_PATH = _PARSED_DIRECTORY.joinpath(_PARSED_LABELS_CSV_FILENAME)

_MAMMAL_DETECTIONS_XLSX_FILENAME = "INANIN_Deployment2_MM_Detections.xlsx"
_IMPULSIVE_NOISE_XSLX_FILENAME = "INANIN_Impulsive_Noise_Sources.xlsx"

_RAW_FILES_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("raw")
MAMMAL_DETECTIONS_PATH = _RAW_FILES_DIRECTORY.joinpath(_MAMMAL_DETECTIONS_XLSX_FILENAME)
IMPULSIVE_NOISE_PATH = _RAW_FILES_DIRECTORY.joinpath(_IMPULSIVE_NOISE_XSLX_FILENAME)

CLUSTER_WORKDIR = pathlib.Path(f"/cluster/work/{getpass.getuser()}")
DATASET_DIRECTORY = CLUSTER_WORKDIR.joinpath("hdd_copy").absolute()
_GLIDER_DATASET_DIRECTORY = DATASET_DIRECTORY.joinpath("GLIDER phase I deployment")
TMP_DATA_DIR = DATASET_DIRECTORY.parent.joinpath("tmp_download_dir")

if not DATASET_DIRECTORY.exists():
    warnings.warn(f"The dataset directory {str(DATASET_DIRECTORY)} does not exist. This can lead to errors and unexpected results.")

if not TMP_DATA_DIR.exists():
    try:
        TMP_DATA_DIR.mkdir(parents=False, exist_ok=False)
    except Exception as ex:
        warnings.warn(f"Temporary dataset download directory does not exist, and could not be created automatically due to the following error", f"[Error No. {ex.errno}] {ex.strerror}")

def list_local_audiofiles():
    return list(TMP_DATA_DIR.glob("**/*.wav")) + list(DATASET_DIRECTORY.glob("**/*.wav"))

AUDIO_FILE_CSV_PATH = _PARSED_DIRECTORY.joinpath("glider_wav_metadata.csv")

POSITIVE_INSTANCE_CLASS_LABEL = True
NEGATIVE_INSTANCE_CLASS_LABEL = False

DATETIME_FORMAT = "%Y.%m.%dT%H:%M:%S.%f"

# Tracking:
TRACKING_DIRECTORY = REPO_DIR.joinpath("tracking")
EXPERIMENTS_TRACKING_DIRECTORY = TRACKING_DIRECTORY.joinpath("results") # pathlib.Path(__file__).parent.joinpath("results")
EXPERIMENTS_FILE = EXPERIMENTS_TRACKING_DIRECTORY.joinpath("results.csv")
LOGGED_AT_COLUMN = "created_at"
COMMAND_AND_ARCUMENTS_COLUMN = "command"
GIT_COMMIT_HASH_COLUMN = "commit"

c1, c2 = 181/255, 241/255
HEADER_BACKGROUND_COLOR = {"red": c1, "green": c1, "blue": c1, "alpha": 1}
ODD_ROW_BACKGROUND_COLOR = {"red": c2, "green": c2, "blue": c2, "alpha": 1}
EVEN_ROW_BACKGROUND_COLOR = {"red": 1, "green": 1, "blue": 1, "alpha": 1}

# Datasets loading
if os.environ.get("VIRTUAL_DATASET_LOADING") is None:
    warnings.warn("Evironment variable VIRTUAL_DATASET_LOADING is missing, using default value False")
VIRTUAL_DATASET_LOADING = os.environ.get("VIRTUAL_DATASET_LOADING").strip().lower() == "true"

# Saver.py
DEFAULT_PARAMETERS_PATH = REPO_DIR.joinpath("models", "parameters").absolute()
if not DEFAULT_PARAMETERS_PATH.exists():
    DEFAULT_PARAMETERS_PATH.mkdir(parents=True, exist_ok=False)

CHECKPOINTS_PATH = REPO_DIR.joinpath("models", "checkpoints").absolute()
if not CHECKPOINTS_PATH.exists():
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=False)

SCOPES = ['https://www.googleapis.com/auth/drive']
SPREADSHEET_ID = os.environ.get(f"{ENV.upper()}_SPREADSHEET_ID")
SHEET_ID = int(os.environ.get(f"{ENV.upper()}_SHEET_ID"))
SHEETS_LOGGING_ENABLED=True

LOCAL_RESULTS_FILENAME = f"{ENV}_results.csv".lower()
LOCAL_RESULTS_PATH = EXPERIMENTS_TRACKING_DIRECTORY.joinpath(LOCAL_RESULTS_FILENAME)