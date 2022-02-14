#!/usr/bin/env python3
'''
    Script to investigate labels with no associated files.
'''
import pathlib
from datetime import timedelta
import sys

import pandas as pd 
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

file_list = pd.read_csv(config.AUDIO_FILE_CSV_PATH)
file_list["end_time"] = pd.to_datetime(file_list["end_time"], format=config.DATETIME_FORMAT, errors="coerce")
file_list["start_time"] = pd.to_datetime(file_list["start_time"], format=config.DATETIME_FORMAT, errors="coerce")

labels = pd.read_excel(config.MAMMAL_DETECTIONS_PATH, sheet_name="Positive_Detection_Minutes")
labels["datetime"] = pd.to_datetime(labels["datetime"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
class_columns = ['FinWhales', 'PotentialSeiWhales_DistantAirguns', 'Unid.Mysticetes', 'Blackfish', 'Unid.Delphinids', 'SpermWhales']
with_detections = labels[~labels[class_columns].isnull().all(axis=1)]

for index in with_detections.index:
    detection = with_detections.loc[index]
    start_time = detection["datetime"]
    end_time = start_time + timedelta(minutes=1)
    overlapping_files = file_list[(file_list["start_time"] <= end_time) & (file_list["end_time"] >= start_time)]
    print(detection, overlapping_files["filename"].values)