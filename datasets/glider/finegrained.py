#!/usr/bin/env python3
import pathlib
import sys
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
from tracking.logger import Logger, LogFormatter, BasicLogger

def parse():
    raw = pd.read_csv(config.RAW_FINEGRAINED_LABELS_PATH)
    raw.drop_duplicates(keep="first", inplace=True)
    cols_to_keep = ["start_time","end_time","source_class","source_class_specific"]
    raw[cols_to_keep].to_csv(config.FINE_GRAINED_LABELS_PATH, sep=",", index=False)

class FineGrainedClippedDataset(ICustomDataset):
    def __init__(
        self,
        logger_factory: ILoggerFactory,
        worker: IAsyncWorker,
        clip_duration_seconds: float = None, 
        clip_overlap_seconds: float = None, 
        clip_nsamples: int = None, 
        overlap_nsamples: int = None) -> None:

        super().__init__()
        
        self.worker = worker
        self.logger = logger_factory.create_logger()
        self.clip_duration_seconds = clip_duration_seconds
        self.clip_overlap_seconds = clip_overlap_seconds
        self.clip_nsamples = clip_nsamples
        self.overlap_nsamples = overlap_nsamples
        
        self.labels = pd.read_csv(config.FINE_GRAINED_LABELS_PATH)
        self.labels.start_time = pd.to_datetime(self.labels.start_time, errors="coerce")
        self.labels.end_time = pd.to_datetime(self.labels.end_time, errors="coerce")

        print(self.labels)
        label_durations = (self.labels.end_time - self.labels.start_time).dt.seconds
        print("Max label dur:", label_durations.max())
        print("Mean label duration:", label_durations.mean(), label_durations.std(), label_durations.quantile(q=0.9))

        old_labels = pd.read_csv(config.PARSED_LABELS_PATH)
        old_labels.start_time = pd.to_datetime(old_labels.start_time, errors="coerce")
        old_labels.end_time = pd.to_datetime(old_labels.end_time, errors="coerce")
        olddur = (old_labels.end_time - old_labels.start_time).dt.seconds
        print("Mean label durations for old labels:", olddur.mean(), olddur.std(), olddur.quantile(q=0.9))

        self.audiofiles = pd.read_csv(config.AUDIO_FILE_CSV_PATH)
        self.audiofiles.start_time = pd.to_datetime(self.audiofiles.start_time, errors="coerce")
        self.audiofiles.end_time = pd.to_datetime(self.audiofiles.end_time, errors="coerce")
        print(self.audiofiles)
        file_durations = (self.audiofiles.end_time - self.audiofiles.start_time).dt.seconds

        print("Mean file duration:", file_durations.mean(), file_durations.std(), file_durations.quantile(q=0.9))

        import matplotlib.pyplot as plt
        print(len(self.labels), len(old_labels))
        overlapping = []
        no_overlapping = []
        for i in range(len(self.labels)):
            newlab = self.labels.iloc[i]

            overlapping_old_labels = old_labels[(old_labels.start_time <= newlab.end_time) & (old_labels.end_time >= newlab.start_time)]
            if len(overlapping_old_labels) > 0:
                overlapping += list(overlapping_old_labels.index.values)
            else:
                no_overlapping.append(i)

        dup = old_labels.iloc[overlapping]
        print(dup)
        dup.drop_duplicates(subset=["start_time", "end_time", "source_class"], keep="first", inplace=True)
        print(dup)
        print(old_labels)
        old_labels.drop_duplicates(subset=["start_time", "end_time", "source_class"], keep="first", inplace=True)
        print(old_labels)

        print(f"New labels overlap with approx.: {(len(dup) / len(old_labels))*100:.2f}% of the old labels (out of {len(old_labels)} old labels, {len(dup)} of them overlap with at least one of the new labels. Keep in mind that this does not take into account how much of the labels overlap, just that they do in fact overlap.")

        not_seen_before = self.labels.iloc[no_overlapping]
        print(not_seen_before)
        print(f"Out of {len(self.labels)} there are {len(not_seen_before)} previously unseen labels (e.g. they have no overlapping labels in the old set)")

        bins=100
        plt.subplot(2, 1, 1)
        plt.title("New labels duration historgram")
        plt.hist(label_durations.values, bins=bins, label=f"{len(self.labels)} new labels")
        plt.ylabel("Number of new labels in bin")
        plt.xlabel("Duration (seconds)")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Old labels duration historgram")
        plt.hist(olddur.values, bins=bins, label=f"{len(old_labels)} old labels")
        plt.ylabel("Number of old labels in bin")
        plt.xlabel("Duration (seconds)")
        plt.legend()
        # plt.subplot(3, 1, 3)
        # plt.hist(file_durations.values, bins=20, density=True)
        plt.show()
        exit()
        valid_audiofile_indeces = []
        for i in range(len(self.audiofiles)):
            print(i, len(self.audiofiles))
            file = self.audiofiles.iloc[i]
            overlapping = self.labels[(self.labels.start_time <= file.end_time) & (self.labels.end_time >= file.start_time)]
            if len(overlapping) > 0:
                valid_audiofile_indeces.append(i)
        
        print(self.audiofiles)
        self.audiofiles = self.audiofiles.iloc[valid_audiofile_indeces]
        print(self.audiofiles)

    def __getitem__(self, index: int) -> LabeledAudioData:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def classes(self) -> Mapping[str, int]:
        return super().classes()

    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        return super().example_shapes()

    def __repr__(self) -> str:
        return super().__repr__()

if __name__ == "__main__":
    parse()
    from datasets.binjob import Binworker

    logger_factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(LogFormatter(),)
    )

    worker = Binworker()

    dataset = FineGrainedClippedDataset(
        logger_factory=logger_factory,
        worker=worker,
        clip_duration_seconds=10.0, 
        clip_overlap_seconds=4.0
    )
    
        