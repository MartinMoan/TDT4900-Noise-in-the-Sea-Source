#!/usr/bin/env python3
from datetime import datetime
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
from interfaces import ICustomDataset, IAsyncWorker, IAudioFileInfoProvider
from cacher import Cacher
from audiodata import LabeledAudioData
from datasets.binjob import progress
from datasets.glider.wavfileinspector import WaveFileHeader, WaveFileInspector
from datasets.glider.fileinfo import AudioFileInfoProvider
from datasets.glider.filelist import AudioFileListProvider
from datasets.binjob import Binworker

class CachedClippedDataset(ICustomDataset):
    def __new__(
        cls, 
        worker: IAsyncWorker, 
        clip_duration_seconds: float = None, 
        clip_overlap_seconds: float = None, 
        clip_nsamples: int = None, 
        overlap_nsamples: int = None,
        force_recache=False) -> ICustomDataset:

        cacher = Cacher()
        hashable_arguments = dict(
            clip_duration_seconds=clip_duration_seconds,
            clip_overlap_seconds=clip_overlap_seconds,
            clip_nsamples=clip_nsamples,
            overlap_nsamples=overlap_nsamples,
        )
        init_kwargs = {"worker": worker, **hashable_arguments}
        return cacher.cache(ClippedDataset, init_args=(), init_kwargs=init_kwargs, hashable_arguments=hashable_arguments, force_recache=force_recache)

class ClippedDataset(ICustomDataset):
    def __init__(
        self,
        clip_duration_seconds: float = None, 
        clip_overlap_seconds: float = None, 
        clip_nsamples: int = None, 
        overlap_nsamples: int = None) -> None:

        self.worker = Binworker()

        # self.filelistprovider = AudioFileListProvider()
        # files = self.filelistprovider.list()
        
        # fileinfoprovider = AudioFileInfoProvider(filelist=files, worker=worker)
        # self._audiofiles = fileinfoprovider.files()
        self._audiofiles = pd.read_csv(config.AUDIO_FILE_CSV_PATH)

        self._labels = pd.read_csv(config.PARSED_LABELS_PATH)

        self._preprocess_tabular_data()

        if config.VIRTUAL_DATASET_LOADING:
            warnings.warn("The environment variable VIRTUAL_DATASET_LOADING is set to True, meaning that the GLIDER dataset loading class will only simulate loading datasets from disk. Ensure that this variable is not set during training or inference, as it is only intended to be used during local development.")
            import socket
            if "idun-login" in socket.gethostname():
                raise Exception(f"GLIDER detected that the current hostname ({socket.gethostname()}) seems to correspond to the NTNU Idun computing cluster, while the VIRTUAL_DATASET_LOADING environment variable was set to True. This variable is only intended for local development and can cause unexpected results. This exception is raised to ensure that logs and model parameters are not overwritten using invalid/simulated data.")

            labeled_audiofiles = []
            for labelidx in self._labels.index:
                label = self._labels.iloc[labelidx]
                labeled = self._audiofiles[(self._audiofiles.start_time <= label.end_time) & (self._audiofiles.end_time >= label.start_time)]
                if len(labeled) > 0:
                    labeled_audiofiles += labeled.index.values.tolist()
            
            self._audiofiles = self._audiofiles.loc[labeled_audiofiles, :]
            self._audiofiles = self._audiofiles.iloc[:50]

        sampling_rate = self._audiofiles.sampling_rate.max()
        if clip_duration_seconds is None and clip_nsamples is not None:
            self._clip_duration = clip_nsamples / sampling_rate
        else:
            self._clip_duration = clip_duration_seconds
        
        if clip_overlap_seconds is None and overlap_nsamples is not None:
            self._clip_overlap = overlap_nsamples / sampling_rate
        else:
            self._clip_overlap = clip_overlap_seconds

        self._max_file_duration = self._audiofiles.duration_seconds.max()
        self._num_clips_per_file = math.floor((self._max_file_duration - self._clip_overlap) / (self._clip_duration - self._clip_overlap))
        
        label_values = self._labels.source_class.unique()
        self._classes = {label_values[idx]: idx for idx in range(len(label_values))}

        self._verify_files()

    def _verify(self, virtual_indeces: Iterable[int], start: int, stop: int) -> Tuple[Iterable[int], Iterable[int]]:
        valid_indeces = []
        invalid_indeces = []
        proc = multiprocessing.current_process()
        
        for i in range(start, min(len(virtual_indeces), stop)):
            should_log, percentage = progress(i, start, stop)
            if should_log:
                print(f"ClippingWorker PID {proc.pid} - {percentage:.2f}%")
            try:
                # file_index = math.floor(i / self._num_clips_per_file)
                audiodata = self._load(i)
                if audiodata.start_time >= audiodata.file_end_time or audiodata.end_time > audiodata.file_end_time:
                    invalid_indeces.append(i)
                else:
                    valid_indeces.append(i)
            except Exception as ex:
                print(i, ex)
                invalid_indeces.append(i)
        return valid_indeces, invalid_indeces

    def _agg(self, results: Iterable[Tuple[Iterable[int], Iterable[int]]]) -> Tuple[Iterable[int], Iterable[int]]:
        valid, invalid = [], []
        for valid_indeces, invalid_indeces in results:
            valid += valid_indeces
            invalid += invalid_indeces
        return valid, invalid

    def _verify_files(self):
        valid_indeces, invalid_indeces = self.worker.apply(range(0, self._virtual_length()), self._verify, aggregation_method=self._agg)
        self._valid_clip_indeces = np.sort(valid_indeces, axis=0)
        self._invalid_clip_indeces = np.sort(invalid_indeces, axis=0)
                
    def _preprocess_tabular_data(self):
        # ensure time columns are datetime/np.datetime64 objects
        self._audiofiles.start_time = pd.to_datetime(self._audiofiles.start_time, errors="coerce")
        self._audiofiles.end_time = pd.to_datetime(self._audiofiles.end_time, errors="coerce")

        invalid_datetimes = self._audiofiles[self._audiofiles.start_time.isnull()]
        self._audiofiles = self._audiofiles.drop(index=invalid_datetimes.index, axis="index")
        
        self._labels.start_time = pd.to_datetime(self._labels.start_time, errors="coerce")
        self._labels.end_time = pd.to_datetime(self._labels.end_time, errors="coerce")
        
        # Remove any files from self._audilfiles that cannot be found in local dataset directory
        # and ensure that all filename values in the self._audiofiles dataframe are PosixPath objects. 
        # local_audiofiles = self.filelistprovider.list()
        local_audiofiles = list(config.DATASET_DIRECTORY.glob("**/*.wav"))
        if len(local_audiofiles) == 0:
            raise Exception(f"No audiofiles could be found in dataset directory: {config.DATASET_DIRECTORY}")
        if len(local_audiofiles) != len(self._audiofiles):
            if len(local_audiofiles) > len(self._audiofiles):
                warnings.warn(f"There are local files that do not exist among the described in the csv at {config.PARSED_LABELS_PATH}")
            elif len(local_audiofiles) < len(self._audiofiles):
                warnings.warn(f"There are files described in the csv at {config.PARSED_LABELS_PATH} that do not exist in any subdirectory of {config.DATASET_DIRECTORY}")
        
        self._audiofiles.filename = [config.DATASET_DIRECTORY.joinpath(path) for path in self._audiofiles.filename.values]    
        self._audiofiles = self._audiofiles[self._audiofiles.filename.isin(local_audiofiles)]

        # sort according to start- and end times
        self._audiofiles = self._audiofiles.sort_values(by=["start_time", "end_time"])

    def _get_labels(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self._labels[(self._labels.start_time <= end) & (self._labels.end_time >= start)]    

    def _load(self, virtual_clip_index: int):
        file_index = math.floor(virtual_clip_index / self._num_clips_per_file)

        first_clip_in_file_index = int(file_index * self._num_clips_per_file) # clip index of first clip in file
        nth_clip_in_file = virtual_clip_index - first_clip_in_file_index

        relative_clip_start = nth_clip_in_file * (self._clip_duration - self._clip_overlap)
        # relative_clip_end = ((nth_clip_in_file+1) * (self._clip_duration - self._clip_overlap)) + self._clip_overlap

        file = self._audiofiles.iloc[file_index]
        
        return LabeledAudioData(
            _index = virtual_clip_index,
            filepath = file.filename,
            num_channels = file.num_channels,
            sampling_rate = file.sampling_rate,
            file_start_time = file.start_time,
            file_end_time = file.end_time,
            clip_duration = self._clip_duration,
            clip_offset = relative_clip_start,
            all_labels = self._labels,
            labels_dict = self._classes
        )

    def __getitem__(self, clip_index: int):
        virtual_clip_index = self._valid_clip_indeces[clip_index]
        return self._load(virtual_clip_index)

    def __len__(self) -> int:
        return len(self._valid_clip_indeces)

    def _virtual_length(self) -> int: 
        return int(self._num_clips_per_file * len(self._audiofiles))

    def classes(self) -> Mapping[str, int]:
        return self._classes

    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        sr = self._audiofiles.sampling_rate.max()
        num_samples = int(self._clip_duration * sr)
        num_channels = self._audiofiles.num_channels.max()
        return [(num_channels, num_samples) for _ in range(len(self))]

    def __repr__(self):
        relevant_attributes = ["_clip_duration", "_clip_overlap", "_classes", "_max_file_duration", "_num_clips_per_file", "_valid_clip_indeces"]
        relevant_values = {key: getattr(self, key) for key in relevant_attributes}
        out = f"{self.__class__.__name__}( {repr(relevant_values)} )"
        return repr(out)

    def __str__(self):
        relevant_attributes = ["_clip_duration", "_clip_overlap", "_classes", "_max_file_duration", "_num_clips_per_file", "_valid_clip_indeces"]
        relevant_values = {key: str(getattr(self, key)) for key in relevant_attributes}
        out = f"{self.__class__.__name__}( {str(relevant_values)} )"
        return str(out)

if __name__ == "__main__":
    from datasets.binjob import Binworker

    dataset = CachedClippedDataset(
        clip_duration_seconds=10.0, 
        clip_overlap_seconds=4.0
    )
    
        
        