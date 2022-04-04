#!/usr/bin/env python3
from datetime import datetime, timedelta
import pathlib
import sys
import math
import warnings
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
from typing import Iterable, Mapping

from rich import print
import pandas as pd
import git
sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
from ITensorAudioDataset import MelSpectrogramFeatureAccessor

class ClippedDataset(ICustomDataset):
    def __init__(self, clip_duration_seconds = None, clip_overlap_seconds = None, clip_nsamples: int = None, overlap_nsamples = None) -> None:
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

    def _verify(self, start: int, stop: int):
        valid_indeces = []
        invalid_indeces = []
        proc = multiprocessing.current_process()
        
        for i in range(start, min(stop, self._virtual_length())):
            percentage = ((i - start) / (stop - start)) * 100
            part = math.ceil((stop - start) * 0.025)
            if (i - start) % part == 0:
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

    def _verify_files(self):
        n_processes = multiprocessing.cpu_count()
        with Pool(processes=n_processes) as pool:
            bin_size = math.ceil(self._virtual_length() / n_processes)
            
            bins = [(start, (start + bin_size)) for start in range(0, self._virtual_length(), bin_size)]

            tasks = []
            for bin in bins:
                start, stop = bin
                tasks.append(pool.apply_async(self._verify, (start, stop)))
            
            valid_indeces_results = []
            invalid_indeces_results = []
            for task in tasks:
                valid_indeces, invalid_indeces = task.get()
                valid_indeces_results += valid_indeces
                invalid_indeces_results += invalid_indeces

            self._valid_clip_indeces = valid_indeces_results
            self._invalid_clip_indeces = invalid_indeces_results
                
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
        local_audiofiles = config.list_local_audiofiles()
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

if __name__ == "__main__":
    dataset = ClippedDataset(clip_duration_seconds=10.0, clip_overlap_seconds=4.0)
    from matplotlib import pyplot as plt
    spectrogram_computer = MelSpectrogramFeatureAccessor(n_mels=128, n_fft=2048, hop_length=512)
    for i in range(len(dataset)):
        audiodata = dataset[i]
        print(audiodata)
        labels = audiodata.labels
        print(labels)
        if len(labels) > 0:
            spect = spectrogram_computer(audiodata)
            spect = spect.squeeze()
            classes = ", ".join(labels.source_class_specific.unique())
            plt.suptitle(audiodata.filepath.name)
            plt.title(classes) 
            plt.imshow(spect, aspect="auto", extent=[0, spect.shape[1], 0, spect.shape[0]])
            plt.show()