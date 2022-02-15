#!/usr/bin/env python3
import pathlib
import sys
import os
import warnings

import dotenv
dotenv.load_dotenv()

import git
from rich import print

repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
REPO_DIR = pathlib.Path(repo.working_dir).absolute()

for directory in [d for d in REPO_DIR.glob("**/*") if d.is_dir()]:
    sys.path.insert(0, str(directory))

_GLIDER_DATASET_LABELS_DIRECTORY = REPO_DIR.joinpath("datasets", "glider")

_PARSED_LABELS_CSV_FILENAME = "glider_labels.csv"
_PARSED_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("parsed")
PARSED_LABELS_PATH = _PARSED_DIRECTORY.joinpath(_PARSED_LABELS_CSV_FILENAME)

_MAMMAL_DETECTIONS_XLSX_FILENAME = "INANIN_Deployment2_MM_Detections.xlsx"
_IMPULSIVE_NOISE_XSLX_FILENAME = "INANIN_Impulsive_Noise_Sources.xlsx"

_RAW_FILES_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("raw")
MAMMAL_DETECTIONS_PATH = _RAW_FILES_DIRECTORY.joinpath(_MAMMAL_DETECTIONS_XLSX_FILENAME)
IMPULSIVE_NOISE_PATH = _RAW_FILES_DIRECTORY.joinpath(_IMPULSIVE_NOISE_XSLX_FILENAME)

DATASET_DIRECTORY = pathlib.Path("/cluster/work/martimoa/hdd_copy/").absolute()

_AUDIO_FILE_LIST = list(DATASET_DIRECTORY.glob("**/*.wav"))
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

# SCOPES = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
SCOPES = ['https://www.googleapis.com/auth/drive']
SPREADSHEET_ID = '1qT3gS0brhu2wj59cyeZYP3AywGErROJCqR2wYks6Hcw'
SHEET_ID = 267619714

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