#!/usr/bin/env python3
from datetime import datetime
from email.mime import audio
import pathlib
import sys
import math
import warnings
import multiprocessing
from typing import Iterable, Mapping, Tuple

from rich import print
import pandas as pd
import git
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ICustomDataset, IAsyncWorker, ILoggerFactory, IAudioFileInfoProvider
from cacher import Cacher
from audiodata import LabeledAudioData
from datasets.binjob import progress
from datasets.glider.wavfileinspector import WaveFileHeader, WaveFileInspector
from datasets.glider.fileinfo import AudioFileInfoProvider
from datasets.glider.filelist import AudioFileListProvider
from tracking.loggerfactory import LoggerFactory
from tracking.logger import Logger, LogFormatter


def main():
    audiofiles = pd.read_csv(config.AUDIO_FILE_CSV_PATH)
    labels = pd.read_csv(config.PARSED_LABELS_PATH)
    for i in range(len(audiofiles)):
        print(i, len(audiofiles))
        file = audiofiles.iloc[i, :]
        # files = all_labels[(all_labels.start_time <= end_time) & (all_labels.end_time >= start_time)]    
        overlapping_labels = labels[(labels.start_time <= file.end_time) & (labels.end_time >= file.start_time)]
        sources = ", ".join(overlapping_labels.source_class_specific.unique())
        audiofiles.loc[i, "source_string"] = sources

    audiofiles.to_csv("labeled_audiofiles.csv")
    print(audiofiles)
    print(labels)

if __name__ == "__main__":
    main()
