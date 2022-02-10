#!/usr/bin/env python3
import pathlib
import sys
import pandas as pd
from torch.utils.data import Dataset

config_path = pathlib.Path(__file__).parent.joinpath("config.py").absolute()
sys.path.insert(0, str(config_path))
import config

class GLIDER(Dataset):
    def __init__(self):
        self._labels = GLIDER._todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
        self._audiofiles = GLIDER._todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))

    def get_files(self, label):
        start_time, end_time = label["start_time"], label["end_time"]
        # s = label["source_class"]
        subset_files = self._audiofiles[(self._audiofiles["start_time"] >= start_time) & (self._audiofiles["end_time"] <= end_time)]
        # paths = [pathlib.Path(path).name for path in subset_files["filename"].values]

        return subset_files
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)

    def _todatetime(df):
        for col in df.columns:
            if "time" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df


if __name__ == "__main__":
    dataset = GLIDER()

