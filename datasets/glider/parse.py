#!/usr/bin/env python3
import pathlib
import sys

import pandas as pd

config_path = pathlib.Path(__file__).parent.joinpath("config.py").absolute()
sys.path.insert(0, str(config_path))
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
    baleen_whale_encounters["start_time"] = _parse_timestamps(baleen_whale_encounters, "Start time", date_col="Date")
    baleen_whale_encounters["end_time"] = _parse_timestamps(baleen_whale_encounters, "End time", date_col="Date")

    biophonic_idx = baleen_whale_encounters["Anthr. noise interference"].isna()
    biophonic = baleen_whale_encounters[biophonic_idx][["start_time", "end_time", "Delta time", "Detection method", "Species", "Extra observations", "Note:"]]
    anthropogenic = baleen_whale_encounters[~biophonic_idx][["start_time", "end_time", "Delta time", "Detection method", "Anthr. noise interference", "Extra observations", "Note:"]]
    
    anthropogenic["source_class"] = "Anthropogenic"
    biophonic["source_class"] = "Biophonic"

    detection_events.append(anthropogenic)
    detection_events.append(biophonic)

    return detection_events


def parse_toothed_whale_encounters(detection_events):
    toothed_whale_encounters["start_time"] = _parse_timestamps(toothed_whale_encounters, "Start time", date_col="Date")
    toothed_whale_encounters["end_time"] = _parse_timestamps(toothed_whale_encounters, "End time", date_col="Date")

    biophonic_idx = toothed_whale_encounters["Anthr. noise interference"].isna()
    biophonic = toothed_whale_encounters[biophonic_idx][["start_time", "end_time", "Delta time", "Detection method", "Species", "Extra observations"]]
    anthropogenic = toothed_whale_encounters[~biophonic_idx][["start_time", "end_time", "Delta time", "Detection method", "Anthr. noise interference", "Extra observations"]]

    anthropogenic["source_class"] = "Anthropogenic"
    biophonic["source_class"] = "Biophonic"

    detection_events.append(anthropogenic)
    detection_events.append(biophonic)    
     
    return detection_events

def parse_impulsive_noise(detection_events):
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
    detection_events.append(parsed)
    return detection_events
    
def combine_data(detection_events):
    required_columns = ["start_time", "end_time", "source_class"]
    output_cols = ["start_time", "end_time", "source_class", "metadata"]
    combined = pd.DataFrame(columns=output_cols)

    for df in detection_events:
        metadata_df = df.loc[:, list(set(df.columns) - set(required_columns))]
        metadata = []
        for i in range(len(metadata_df)):
            row = metadata_df.iloc[i]
            meta = {col: row[col] for col in metadata_df.columns}
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