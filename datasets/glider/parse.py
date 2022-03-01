#!/usr/bin/env python3
from datetime import timedelta
import pathlib
import sys
import json
import copy 

import pandas as pd
import numpy as np
import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

# positive_detection_minutes = pd.read_excel(config.MAMMAL_DETECTIONS_PATH, sheet_name="Positive_Detection_Minutes")

def _parse_timestamps(df, time_col, date_col="Date"):
    dates = df[date_col].astype(str)
    times = df[time_col].astype(str)
    formated = dates.str.cat(times, sep=" ")
    dt = pd.to_datetime(formated, format=config.DATETIME_FORMAT)
    return dt

def _ensure_start_before_end(df, start_time_col, end_time_col):
    wrong_end_time = df[(df[start_time_col] >= df[end_time_col])]
    df.loc[wrong_end_time.index, end_time_col] += np.timedelta64(1, "D")
    return df

def parse_baleen_whale_encounter(detection_events):
    baleen_whale_encounters = pd.read_excel(config.MAMMAL_DETECTIONS_PATH, sheet_name="Baleen_Whale_Encounters", dtype=object)
    baleen_whale_encounters["source_sheet"] = "Baleen_Whale_Encounters"
    baleen_whale_encounters["sheet_row_id"] = baleen_whale_encounters.index + 2
    baleen_whale_encounters["source_file"] = config.MAMMAL_DETECTIONS_PATH.name
    
    baleen_whale_encounters["start_time"] = _parse_timestamps(baleen_whale_encounters, "Start time", date_col="Date")
    baleen_whale_encounters["end_time"] = _parse_timestamps(baleen_whale_encounters, "End time", date_col="Date")

    baleen_whale_encounters = _ensure_start_before_end(baleen_whale_encounters, "start_time", "end_time")

    anthropogenic_idx = baleen_whale_encounters["Anthr. noise interference"].notna()
    baleen_whale_encounters["source_class"] = "Biophonic"
    baleen_whale_encounters["source_class_specific"] = baleen_whale_encounters["Species"]
    anthropogenic = copy.copy(baleen_whale_encounters.loc[anthropogenic_idx])
    anthropogenic["source_class"] = "Anthropogenic"
    anthropogenic["source_class_specific"] = anthropogenic["Anthr. noise interference"]
    helper = pd.concat([baleen_whale_encounters, anthropogenic], ignore_index=True)
    baleen_whale_encounters = helper
    
    detection_events.append(baleen_whale_encounters)
    return detection_events


def parse_toothed_whale_encounters(detection_events):
    toothed_whale_encounters = pd.read_excel(config.MAMMAL_DETECTIONS_PATH, sheet_name="Toothed_Whale_Encounters")
    toothed_whale_encounters["source_sheet"] = "Toothed_Whale_Encounters"
    toothed_whale_encounters["sheet_row_id"] = toothed_whale_encounters.index + 2
    toothed_whale_encounters["source_file"] = config.MAMMAL_DETECTIONS_PATH.name
    
    toothed_whale_encounters["start_time"] = _parse_timestamps(toothed_whale_encounters, "Start time", date_col="Date")
    toothed_whale_encounters["end_time"] = _parse_timestamps(toothed_whale_encounters, "End time", date_col="Date")

    toothed_whale_encounters = _ensure_start_before_end(toothed_whale_encounters, "Start time", "End time")

    anthropogenic_idx = toothed_whale_encounters["Anthr. noise interference"].notna()
    toothed_whale_encounters["source_class"] = "Biophonic"
    toothed_whale_encounters["source_class_specific"] = toothed_whale_encounters["Species"]
    anthropogenic = copy.copy(toothed_whale_encounters.loc[anthropogenic_idx])
    anthropogenic["source_class"] = "Anthropogenic"
    anthropogenic["source_class_specific"] = anthropogenic["Anthr. noise interference"]
    helper = pd.concat([toothed_whale_encounters, anthropogenic], ignore_index=True)
    toothed_whale_encounters = helper

    detection_events.append(toothed_whale_encounters)     
    return detection_events

def parse_impulsive_noise(detection_events):
    impulsive_noise = pd.read_excel(config.IMPULSIVE_NOISE_PATH, sheet_name="ImpulsiveNoise")
    date_as_string = impulsive_noise["Date"].astype(str)
    starttime_as_string = "T" + impulsive_noise["StartTime"].astype(str)
    start_datetime_as_string = date_as_string.str.cat(starttime_as_string)
    start_datetime = pd.to_datetime(start_datetime_as_string)

    endtime_as_string = "T" + impulsive_noise["EndTime"].astype(str)
    end_datetime_as_string = date_as_string.str.cat(endtime_as_string)
    endtime_datetime = pd.to_datetime(end_datetime_as_string)

    parsed = pd.DataFrame(data={"start_time": start_datetime, "end_time": endtime_datetime})
    parsed = _ensure_start_before_end(parsed, "start_time", "end_time")
    parsed["source_class"] = "Anthropogenic"
    parsed["source_class_specific"] = impulsive_noise["Potential source"]
    
    parsed["source_sheet"] = "ImpulsiveNoise"
    parsed["sheet_row_id"] = impulsive_noise.index + 2
    parsed["source_file"] = config.IMPULSIVE_NOISE_PATH.name

    detection_events.append(parsed)
    return detection_events
    
def combine_data(detection_events):
    required_columns = ["start_time", "end_time", "source_class", "source_class_specific"]
    output_cols = ["start_time", "end_time", "source_class", "metadata"]
    combined = pd.DataFrame(columns=output_cols)

    for df in detection_events:
        if not all(col in df.columns for col in ["source_file", "source_sheet", "sheet_row_id"]):
            print("Df is missing columns")
            print(df.columns.values)

        metadata_df = df.loc[:, list(set(df.columns) - set(required_columns))]
        for col in metadata_df.columns:
            if metadata_df[col].dtype == "datetime64[ns]":
                metadata_df[col] = metadata_df[col].dt.strftime(config.DATETIME_FORMAT)
            elif metadata_df[col].dtype == "int64":
                metadata_df[col] = metadata_df[col].astype(str)
            else: # metadata_df[col].dtype == "object":
                metadata_df[col] = metadata_df[col].astype(str)
        
        metadata = []
        for i in range(len(metadata_df)):
            row = metadata_df.iloc[i]
            d = {col: row[col] for col in metadata_df.columns}
            meta = json.dumps(d)
            metadata.append(meta)
        
        data = pd.DataFrame(df[required_columns].values, columns=required_columns)
        data["metadata"] = metadata
        df = pd.concat([combined, data], ignore_index=True)
        combined = df
    return combined

def main():
    detection_events = []
    detection_events = parse_baleen_whale_encounter(detection_events)
    detection_events = parse_toothed_whale_encounters(detection_events)
    detection_events = parse_impulsive_noise(detection_events)
    combined = combine_data(detection_events)
    print(combined)
    combined.to_csv(config.PARSED_LABELS_PATH, index=False)

if __name__ == "__main__":
    main()