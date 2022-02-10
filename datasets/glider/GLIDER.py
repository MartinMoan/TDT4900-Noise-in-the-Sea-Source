#!/usr/bin/env python3
import pathlib
import sys

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from rich import print

config_path = pathlib.Path(__file__).parent.joinpath("config.py").absolute()
sys.path.insert(0, str(config_path))
import config

class GLIDER(Dataset):
    def __init__(self):
        self._labels = GLIDER._todatetime(pd.read_csv(config.PARSED_LABELS_PATH))
        self._audiofiles = GLIDER._todatetime(pd.read_csv(config.AUDIO_FILE_CSV_PATH))
        print(self._audiofiles)
        print(self._labels)
        unique_labels = self._labels.source_class.unique()
        for label in unique_labels:
            self._audiofiles[label] = np.nan
        n_labels = len(unique_labels)

        for idx in self._audiofiles.index:
            row = self._audiofiles.loc[idx]
            file_labels = self._labels[(self._labels.start_time >= row.start_time) & (self._labels.end_time <= row.end_time)]
            for label in file_labels.source_class.values:
                self._audiofiles.loc[idx, label] = 1

        print(self._audiofiles)

        print(self._audiofiles[(~self._audiofiles["Anthropogenic"].isna())].filename.values)

        
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

