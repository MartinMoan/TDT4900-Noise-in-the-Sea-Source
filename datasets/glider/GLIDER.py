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
        self._labels = pd.read_csv(config.PARSED_LABELS_PATH)
        self._audiofiles = config._AUDIO_FILE_LIST

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)


if __name__ == "__main__":
    dataset = GLIDER()

