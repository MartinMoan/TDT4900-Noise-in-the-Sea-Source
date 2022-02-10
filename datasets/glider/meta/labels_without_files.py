#!/usr/bin/env python3
import pathlib
import sys
import json

import pandas as pd
import numpy as np
from rich import print

config_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(config_path))
import config

def _todatetime(df):
    for col in df.columns:
        if "time" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def get_files(audiofiles_df, label):
        start_time, end_time = label["start_time"], label["end_time"]
        subset_files = audiofiles_df[(audiofiles_df["start_time"] >= start_time) & (audiofiles_df["end_time"] <= end_time)]
        return subset_files

def main():
    _labels = _todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
    _audiofiles = _todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))

    labels_without_files = {"source_file": [], "sheet_row_id": [], "source_sheet": [], "start_time": [], "end_time": [], "Detection method": []}
    for i in _labels.index:
        row = _labels.iloc[i]
        files = get_files(_audiofiles, row)
        if len(files) == 0:
            metadata = json.loads(row["metadata"])
            print(metadata)
            labels_without_files["source_file"].append(metadata["source_file"])
            labels_without_files["sheet_row_id"].append(metadata["sheet_row_id"])
            labels_without_files["source_sheet"].append(metadata["source_sheet"])
            labels_without_files["start_time"].append(row["start_time"])
            labels_without_files["end_time"].append(row["end_time"])
            if "Detection method" in metadata.keys():
                labels_without_files["Detection method"].append(metadata["Detection method"])
            else:
                labels_without_files["Detection method"].append(np.nan)
                
    missing = pd.DataFrame(data=labels_without_files)
    sheet_name = "LabelsWithoutOverlapingFiles"
    writer = pd.ExcelWriter(pathlib.Path(__file__).parent.joinpath("missing_labels.xlsx"))
    missing.to_excel(writer, sheet_name=sheet_name, index=False, na_rep="NaN")
    for column in missing.columns:
        width = max(missing[column].astype(str).map(len).max(), len(column))
        col_idx = missing.columns.get_loc(column)
        writer.sheets[sheet_name].set_column(col_idx, col_idx, width)

    files = _audiofiles[["filename","num_channels","sampling_rate","num_samples","duration_seconds","start_time","end_time"]]
    file_sheet_name = "GliderDatasetAudioFileList"
    files.to_excel(writer, sheet_name=file_sheet_name, index=False, na_rep="NaN")
    for column in files.columns:
        width = max(files[column].astype(str).map(len).max(), len(column))
        col_idx = files.columns.get_loc(column)
        writer.sheets[file_sheet_name].set_column(col_idx, col_idx, width)

    writer.save()

if __name__ == "__main__":
    main()