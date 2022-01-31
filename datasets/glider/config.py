#!/usr/bin/env python3
import pathlib
import subprocess
import re

import pandas as pd

_GLIDER_DATASET_LABELS_DIRECTORY = pathlib.Path(__file__).parent.absolute()

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