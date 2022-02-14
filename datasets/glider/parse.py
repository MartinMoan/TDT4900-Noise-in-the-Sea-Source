#!/usr/bin/env python3
import pathlib
import sys
import json

import pandas as pd
import numpy as np
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

output_file = config.PARSED_LABELS_PATH
mammal_detections_path = config.MAMMAL_DETECTIONS_PATH
impulsive_noise_detections_path = config.IMPULSIVE_NOISE_PATH

positive_detection_minutes = pd.read_excel(mammal_detections_path, sheet_name="Positive_Detection_Minutes")
baleen_whale_encounters = pd.read_excel(mammal_detections_path, sheet_name="Baleen_Whale_Encounters", dtype=object)
toothed_whale_encounters = pd.read_excel(mammal_detections_path, sheet_name="Toothed_Whale_Encounters")
impulsive_noise = pd.read_excel(impulsive_noise_detections_path, sheet_name="ImpulsiveNoise")

def _parse_timestamps(df, time_col, date_col="Date"):
    datetime_format = f"%Y-%m-%d %H:%M:%f"
    dates = df[date_col].astype(str)
    times = df[time_col].astype(str)
    formated = dates.str.cat(times, sep=" ")
    dt = pd.to_datetime(formated, format=datetime_format)
    return dt

def parse_baleen_whale_encounter(detection_events):
    baleen_whale_encounters["source_sheet"] = "Baleen_Whale_Encounters"
    baleen_whale_encounters["sheet_row_id"] = baleen_whale_encounters.index + 2
    baleen_whale_encounters["source_file"] = mammal_detections_path.name
    
    baleen_whale_encounters["start_time"] = _parse_timestamps(baleen_whale_encounters, "Start time", date_col="Date")
    baleen_whale_encounters["end_time"] = _parse_timestamps(baleen_whale_encounters, "End time", date_col="Date")

    biophonic_idx = baleen_whale_encounters["Anthr. noise interference"].isna()
    
    baleen_whale_encounters.loc[biophonic_idx, "source_class"] = "Biophonic"
    baleen_whale_encounters.loc[~biophonic_idx, "source_class"] = "Anthropogenic"

    detection_events.append(baleen_whale_encounters)

    return detection_events


def parse_toothed_whale_encounters(detection_events):
    toothed_whale_encounters["source_sheet"] = "Toothed_Whale_Encounters"
    toothed_whale_encounters["sheet_row_id"] = toothed_whale_encounters.index + 2
    toothed_whale_encounters["source_file"] = mammal_detections_path.name
    
    toothed_whale_encounters["start_time"] = _parse_timestamps(toothed_whale_encounters, "Start time", date_col="Date")
    toothed_whale_encounters["end_time"] = _parse_timestamps(toothed_whale_encounters, "End time", date_col="Date")

    biophonic_idx = toothed_whale_encounters["Anthr. noise interference"].isna()

    toothed_whale_encounters.loc[biophonic_idx, "source_class"] = "Anthropogenic"
    toothed_whale_encounters.loc[~biophonic_idx, "source_class"] = "Biophonic"

    detection_events.append(toothed_whale_encounters)
     
    return detection_events

def parse_impulsive_noise(detection_events):
    toothed_whale_encounters["source_sheet"] = "Toothed_Whale_Encounters"
    toothed_whale_encounters["sheet_row_id"] = toothed_whale_encounters.index + 2
    toothed_whale_encounters["source_file"] = mammal_detections_path.name
    
    date_as_string = impulsive_noise["Date"].astype(str)
    starttime_as_string = "T" + impulsive_noise["StartTime"].astype(str)
    start_datetime_as_string = date_as_string.str.cat(starttime_as_string)
    start_datetime = pd.to_datetime(start_datetime_as_string)

    endtime_as_string = "T" + impulsive_noise["EndTime"].astype(str)
    end_datetime_as_string = date_as_string.str.cat(endtime_as_string)
    endtime_datetime = pd.to_datetime(end_datetime_as_string)

    parsed = pd.DataFrame(data={"start_time": start_datetime, "end_time": endtime_datetime})
    parsed["source_class"] = "Anthropogenic"
    parsed["Potential source"] = impulsive_noise["Potential source"]
    
    parsed["source_sheet"] = "ImpulsiveNoise"
    parsed["sheet_row_id"] = impulsive_noise.index + 2
    parsed["source_file"] = impulsive_noise_detections_path.name

    detection_events.append(parsed)
    return detection_events
    
def combine_data(detection_events):
    required_columns = ["start_time", "end_time", "source_class"]
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
        combined = combined.append(data, ignore_index=True)
    return combined

def main():
    detection_events = []
    detection_events = parse_baleen_whale_encounter(detection_events)
    detection_events = parse_toothed_whale_encounters(detection_events)
    detection_events = parse_impulsive_noise(detection_events)
    combined = combine_data(detection_events)
    print(combined)
    combined.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()