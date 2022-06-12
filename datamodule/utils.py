#!/usr/bin/env python3
import pickle
import datetime
import multiprocessing
import pathlib
from tabnanny import filename_only
from typing import Iterable, Optional, Union, Dict
from multiprocessing.pool import ThreadPool
import hashlib

import re
import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm
import librosa

def typecheck_audiofile_directory(audiofile_directory):
    audiofile_directories = []
    if isinstance(audiofile_directory, (np.ndarray, list, tuple)):
        for path in audiofile_directory:
            if isinstance(path, pathlib.Path):
                audiofile_directories.append(path)
            elif isinstance(path, str):
                path = pathlib.Path(path)
                audiofile_directories.append(pathlib.Path(path))
            else:
                raise TypeError(f"audiofile path value has invalid type, expected string or pathlib.Path but received {type(path)}")
    elif isinstance(audiofile_directory, pathlib.Path):
        audiofile_directories = [audiofile_directory]
    elif isinstance(audiofile_directory, str):
        audiofile_directories = [pathlib.Path(audiofile_directory)]
    else:
        raise TypeError(f"audiofile_directory has incorrect type {type(audiofile_directory)}")
    return audiofile_directories

def ensure_dirs_exists(directories: Iterable[Union[pathlib.Path, str]]) -> None:
    for dir in directories:
        path = pathlib.Path(dir)
        if not path.exists():
            raise ValueError(f"Path {path} does not exists")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
    return directories

def ensure_dataframe_columns(df: pd.DataFrame) -> None:    
    required_columns = ["source_class", "start_time", "end_time"]
    for column in required_columns:
        if column not in df.columns:
            raise Exception(f"labels DataFrame is missing required column {column}")
    df.start_time = pd.to_datetime(df.start_time, errors="coerce")
    df.end_time = pd.to_datetime(df.end_time, errors="coerce")
    class_values = df.source_class.unique()
    return df, class_values

def list_audiofiles(path: pathlib.Path):
    return list(path.glob("**/*.wav"))

def get_audiofiles(directories: Iterable[pathlib.Path], verbose: bool = False) -> Iterable[pathlib.Path]:
    output = []
    if verbose:
        print("Finding audiofiles...")
        for dir in tqdm(directories):
            output += list_audiofiles(dir)
    else:
        for dir in directories:
            output += list_audiofiles(dir)
    return output

def parse_audio_filename(path: pathlib.Path) -> Dict[str, Union[datetime.datetime, pathlib.Path, Exception]]:
    filename_information = re.search(r"([^_]+)_([0-9]{3})_([0-9]{6}_[0-9]{6})\.wav", path.name)
    # dive_number = filename_information.group(1)
    # identification_number = filename_information.group(2)
    timestring = filename_information.group(3)
    start_time = None
    error = None
    try:
        start_time = datetime.datetime.strptime(timestring, "%y%m%d_%H%M%S")
    except Exception as ex:
        error = ex
        start_time = datetime.datetime(year=1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return dict(
        filepath=path,
        start_time=start_time,
        error=error
    )

def parse_filenames(audiofiles: Iterable[pathlib.Path]) -> Iterable[Dict[str, Union[datetime.datetime, pathlib.Path, Exception]]]:
    return [parse_audio_filename(path) for path in audiofiles]

def get_info(filename_information):
    if filename_information["error"] is not None:
        return None

    samples, sr = librosa.load(filename_information["filepath"], sr=None)
    duration_seconds = len(samples) / sr
    return dict(
        filepath=filename_information["filepath"],
        start_time=filename_information["start_time"],
        end_time=filename_information["start_time"] + datetime.timedelta(seconds=duration_seconds),
        sample_rate=sr,
        num_samples=len(samples),
        duration_seconds=duration_seconds
    )
    
def chunk(iterable, func, position):
    output = []
    with tqdm(total=len(iterable), position=position, leave=False) as pbar:
        for i in range(len(iterable)):
            pbar.update(1)
            try:
                data = func(iterable[i])
                if data is not None:
                    output.append(data)
            except Exception as ex:
                print(f"An exception occured when reading the file {iterable[i]}, skipping file: {ex}")
                continue
        return output

def get_audiofile_information(audiofiles: Iterable[pathlib.Path], processes: int = multiprocessing.cpu_count(), limit: Optional[int] = None) -> pd.DataFrame:
    filename_info = parse_filenames(audiofiles)
    max = limit if limit is not None else len(filename_info)
    chunksize = max / processes
    with ThreadPool(processes=processes) as pool:
        tasks = []
        print("Initializing jobs...")
        for i in range(processes + 1):
            start = int(i*chunksize)
            stop = min(max, int((i+1)*chunksize))
            task = pool.apply_async(chunk, args=(filename_info[start:stop], get_info, i))
            tasks.append(task)
        
        print(f"Performing {len(tasks)} jobs...")
        result = np.array([])
        for task in tasks:
            result = np.concatenate((result, task.get()))
        return pd.DataFrame(data=list(result))

def get_metadata(audiofiles: Iterable[pathlib.Path], cache_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = pathlib.Path(__file__).parent.joinpath("metadata")
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=False)

    sorted_audiofiles = np.sort(audiofiles, axis=0)
    hasher = hashlib.sha256()
    hasher.update(repr(sorted_audiofiles).encode())
    hash = hasher.hexdigest()
    pickle_filename = f"{str(hash)}.pickle"
    pickle_path = cache_dir.joinpath(pickle_filename).absolute()
    cached_metadata_filepaths = [path for path in cache_dir.glob("**/*.pickle")]

    if pickle_path in cached_metadata_filepaths:
        print(f"Loading from cache {pickle_path}")
        with open(pickle_path, "rb") as binary_file:
            return pickle.load(binary_file)
    else:
        output = get_audiofile_information(audiofiles=audiofiles)
        with open(pickle_path, "wb") as binary_file:
            pickle.dump(output, binary_file)
            return output

def ensure_recording_columns(recordings_df):
    required_columns = ["filepath", "start_time", "end_time"]
    for column in required_columns:
        if column not in recordings_df.columns:
            raise Exception(f"labels DataFrame is missing required column {column}")
    recordings_df.start_time = pd.to_datetime(recordings_df.start_time, errors="coerce")
    recordings_df.end_time = pd.to_datetime(recordings_df.end_time, errors="coerce")
    return recordings_df