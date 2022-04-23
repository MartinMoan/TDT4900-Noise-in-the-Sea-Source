#!/usr/bin/env python3
import os
from dataclasses import dataclass
import pathlib
import getpass
import warnings
import dotenv
dotenv.load_dotenv()

import git

repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)

from logger import ILogger, Logger, prettify

@dataclass
class GlobalConfiguration:
    REPO_DIR = pathlib.Path(repo.working_dir).absolute()
    ENV = os.environ.get("ENV", "dev")
    _GLIDER_DATASET_LABELS_DIRECTORY = REPO_DIR.joinpath("datasets", "glider")
    _PARSED_LABELS_CSV_FILENAME = "glider_labels.csv"
    _PARSED_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("parsed")
    _PARSED_DIRECTORY.mkdir(parents=False, exist_ok=True)
    _META_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("meta")
    _META_DIRECTORY.mkdir(parents=False, exist_ok=True)
    _META_MISSING_FILES_DIRECTORY = _META_DIRECTORY.joinpath("missing_files")
    _META_MISSING_FILES_DIRECTORY.mkdir(parents=False, exist_ok=True)
    PARSED_LABELS_PATH = _PARSED_DIRECTORY.joinpath(_PARSED_LABELS_CSV_FILENAME) # used

    _MAMMAL_DETECTIONS_XLSX_FILENAME = "INANIN_Deployment2_MM_Detections.xlsx"
    _IMPULSIVE_NOISE_XSLX_FILENAME = "INANIN_Impulsive_Noise_Sources.xlsx"

    _RAW_FILES_DIRECTORY = _GLIDER_DATASET_LABELS_DIRECTORY.joinpath("raw")
    MAMMAL_DETECTIONS_PATH = _RAW_FILES_DIRECTORY.joinpath(_MAMMAL_DETECTIONS_XLSX_FILENAME) # used
    IMPULSIVE_NOISE_PATH = _RAW_FILES_DIRECTORY.joinpath(_IMPULSIVE_NOISE_XSLX_FILENAME) # used

    CLUSTER_WORKDIR = pathlib.Path(f"/cluster/work/{getpass.getuser()}")
    DATASET_DIRECTORY = CLUSTER_WORKDIR.joinpath("hdd_copy").absolute() # used
    GLIDER_DATASET_DIRECTORY = DATASET_DIRECTORY.joinpath("GLIDER phase I deployment") # used 
    TMP_DATA_DIR = DATASET_DIRECTORY.parent.joinpath("tmp_download_dir") # used

    if not DATASET_DIRECTORY.exists():
        warnings.warn(f"The dataset directory {str(DATASET_DIRECTORY)} does not exist. This can lead to errors and unexpected results.")

    if not TMP_DATA_DIR.exists():
        try:
            TMP_DATA_DIR.mkdir(parents=False, exist_ok=False)
        except Exception as ex:
            warnings.warn(f"Temporary dataset download directory does not exist, and could not be created automatically due to the following error", f"[Error No. {ex.errno}] {ex.strerror}")

    AUDIO_FILE_CSV_PATH = _PARSED_DIRECTORY.joinpath("glider_wav_metadata.csv") # used

    POSITIVE_INSTANCE_CLASS_LABEL = True #used
    NEGATIVE_INSTANCE_CLASS_LABEL = False # used

    DATETIME_FORMAT = "%Y.%m.%dT%H:%M:%S.%f" # used

    # Tracking:
    TRACKING_DIRECTORY = REPO_DIR.joinpath("tracking")
    EXPERIMENTS_TRACKING_DIRECTORY = TRACKING_DIRECTORY.joinpath("results") # pathlib.Path(__file__).parent.joinpath("results")
    EXPERIMENTS_FILE = EXPERIMENTS_TRACKING_DIRECTORY.joinpath("results.csv")
    LOGGED_AT_COLUMN = "created_at" # used

    c1, c2 = 181/255, 241/255
    HEADER_BACKGROUND_COLOR = {"red": c1, "green": c1, "blue": c1, "alpha": 1} # used
    ODD_ROW_BACKGROUND_COLOR = {"red": c2, "green": c2, "blue": c2, "alpha": 1} # used
    EVEN_ROW_BACKGROUND_COLOR = {"red": 1, "green": 1, "blue": 1, "alpha": 1} #used

    # Datasets loading
    if os.environ.get("VIRTUAL_DATASET_LOADING") is None:
        warnings.warn("Evironment variable VIRTUAL_DATASET_LOADING is missing, using default value False")
    VIRTUAL_DATASET_LOADING = os.environ.get("VIRTUAL_DATASET_LOADING").strip().lower() == "true" # used

    SCOPES = ['https://www.googleapis.com/auth/drive'] #used
    SPREADSHEET_ID = os.environ.get(f"{ENV.upper()}_SPREADSHEET_ID") #used
    SHEET_ID = int(os.environ.get(f"{ENV.upper()}_SHEET_ID")) # used
    VERIFICATION_SHEET_ID = int(os.environ.get("VERIFICATION_SHEET_ID")) # used

    CACHING_ENABLED = os.environ.get("CACHING_ENABLED").strip().lower() == "true" # used

    LOCAL_RESULTS_FILENAME = f"{ENV}_results.csv".lower()
    LOCAL_RESULTS_PATH = EXPERIMENTS_TRACKING_DIRECTORY.joinpath(LOCAL_RESULTS_FILENAME)

    HOME_PROJECT_DIR = pathlib.Path.home().joinpath(".nits") #used
    if not HOME_PROJECT_DIR.exists():
        HOME_PROJECT_DIR.mkdir(parents=False, exist_ok=False)
        
    CACHE_DIR = HOME_PROJECT_DIR.joinpath("cache") # used
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=False, exist_ok=False)

    # Saver.py
    DEFAULT_PARAMETERS_PATH = HOME_PROJECT_DIR.joinpath("models", "parameters").absolute() # used
    if not DEFAULT_PARAMETERS_PATH.exists():
        DEFAULT_PARAMETERS_PATH.mkdir(parents=True, exist_ok=False)

    CHECKPOINTS_PATH = HOME_PROJECT_DIR.joinpath("models", "checkpoints").absolute() # used
    if not CHECKPOINTS_PATH.exists():
        CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=False)

    PRINT_INTERVAL_SECONDS = 10 # used

    def list_local_audiofiles(self): 
        return list(self.TMP_DATA_DIR.glob("**/*.wav")) + list(self.DATASET_DIRECTORY.glob("**/*.wav"))

    def __str__(self):
        out = {key: getattr(self, key) for key in dir(self) if (not key.startswith("__") and not key.endswith("__"))}
        return prettify(out)
