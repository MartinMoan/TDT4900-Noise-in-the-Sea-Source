#!/usr/bin/env python3
from os import path
import pathlib
import sys

import pandas as pd
from torch.utils.data import Dataset

config_path = pathlib.Path(__file__).parent.joinpath("config.py").absolute()
sys.path.insert(0, str(config_path))
import config

class GLIDER(Dataset):
    def __init__(self):
        self._labels = pd.read_csv(config.PARSED_LABELS_PATH)
        self._audiofiles = pd.read_csv(config.AUDIO_FILE_CSV_PATH)
        # print(self._audiofiles.sort_values(by=["start_time", "filename"])[["filename", "start_time"]])
        for i in self._labels.index:
            row = self._labels.iloc[i]
            self.get_files(row)

    def get_files(self, label):
        start_time, end_time = label["start_time"], label["end_time"]
        s = label["source_class"]
        subset_files = self._audiofiles[(self._audiofiles["start_time"] >= start_time) & (self._audiofiles["end_time"] <= end_time)]
        paths = [pathlib.Path(path).name for path in subset_files["filename"].values]
        if len(subset_files) == 0:
            print(f"No files exist for label {s} marked with start time {start_time} and end time {end_time}")

        print(len(self._labels))
        # else:
        #     print(f"Found {len(subset_files)} file(s) made withing the starting and ending timestamps for label {s}\nLabel start: {start_time} Label end:{end_time}")

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)


if __name__ == "__main__":
    dataset = GLIDER()

