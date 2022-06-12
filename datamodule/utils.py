#!/usr/bin/env python3
import datetime
import hashlib
import multiprocessing
import pathlib
import pickle
import re
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm


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
    recordings_df.filepath = recordings_df.filepath.apply(lambda filepath: pathlib.Path(filepath))
    recordings_df.start_time = pd.to_datetime(recordings_df.start_time, errors="coerce")
    recordings_df.end_time = pd.to_datetime(recordings_df.end_time, errors="coerce")
    return recordings_df

def label_combinations(class_values: Iterable[str], positive_value: Optional[Any] = True, negative_value: Optional[any] = False):
    C = len(class_values)
    N = 2**C
    helper = np.full((N, C), fill_value=negative_value)
    frequency = 1
    for c in range(len(class_values)):
        current = positive_value
        for i in range(N):
            if i % frequency == 0:
                current = negative_value if current == positive_value else positive_value
            helper[i, c] = current
        frequency *= 2
    output = []
    for i in range(N):
        data = {class_values[c]:  helper[i, c] for c in range(len(class_values))}
        output.append(data)
    return output
import torch
import torch.utils.data


def twopass(dataset, indeces):
    values = np.array([])
    mu = 0
    n = 0
    for i in indeces:
        X, _ = dataset[i]
        x = X.detach().numpy().flatten()
        values = np.concatenate((values, x))
        mu += x.sum()
        n += len(x)

    mu = mu / n
    sigma = 0
    n = 0
    for i in indeces:
        X, _ = dataset[i]
        x = X.detach().numpy().flatten()
        n += len(x)
        sigma += np.sum((x - mu)**2)

    sigma = (sigma / n)**(1/2)
    return mu, sigma

def welford(dataset, indeces):
    mu, sigma, n = 0, 0, 0
    values = np.array([])
    for i in indeces:
        X, _ = dataset[i]
        x = X.detach().numpy().flatten()
        for j in range(len(x)):
            n += 1
            newM = mu + (x[j] - mu) / n
            newS = sigma + (x[j] - mu) * (x[j] - newM)
            mu = newM
            sigma = newS
        values = np.concatenate((values, x))
    
    sigma = (sigma / (n - 1))**(1/2)
    return mu, sigma

def normalize(
    dataset: Union[torch.utils.data.Dataset, Iterable[Tuple[torch.Tensor, torch.Tensor]]], 
    limit: Optional[int] = None, 
    randomize: Optional[Union[float, int, bool]] = True) -> Tuple[float, float]:
    """Compute the arithmetic mean and standard deviation of input data.

    Args:
        dataset (Union[torch.utils.data.Dataset, Iterable[Tuple[torch.Tensor, torch.Tensor]]]): An iterable yielding X, Y (data, target) tuples. Stats are computed from X/data only.
        limit (Optional[int], optional): The maximum number of examples to compute stats from, usefull if dataset is large such that iterating over all examples take a long time. Defaults to None.
        randomize (Optional[int, float, bool], optional): Whether to draw random samples, if float or int will use the value as random seed

    Returns:
        Tuple[float, float]: Arithmetix mean, standard deviation of data
    """
    limit = len(dataset) if limit is None else limit
    
    indeces = np.arange(0, limit, step=1)
    if randomize is not None:
        if isinstance(randomize, (float, int)):
            rng = np.random.default_rng(seed=randomize)
            indeces = rng.integers(0, len(dataset), limit)
        elif isinstance(randomize, bool) and randomize:
            indeces = np.random.randint(0, len(dataset), size=limit)

    mu, sigma, n = 0, 0, 0
    for i in tqdm(indeces):
        X, _ = dataset[i]
        x = X.detach().numpy().flatten()
        n += len(x)
        mu += np.sum(x)
        sigma += np.sum((x**2))
    
    mu = mu / n
    sigma = ((sigma / n) - (mu ** 2))**(1/2)
    return mu, sigma

    """
    Std = (1/N * Var)**(1/2)
    Var = Sum((x - u)**2) = Sum(x**2) - 1/N * Sum(x)**2 = Sum(x**2) - Sum(x)/N * Sum(x) = 
     
    Std = (1/N * (Sum(x**2) - Sum(x)/N * Sum(x)))**(1/2) 
        = (Sum(x**2)/N - Sum(x)/N * Sum(x) * 1/N)**(1/2)
        = (Sum(x**2)/N - (Sum(x)/N)**2)(1/2)
        = (Sum(x**2)/N - u**2)(1/2)
    
    Example:
    (x1 - u)^2 + (x2 - u)^2 + (x3 - u)^2 =
    (x1 - u)(x1 - u) + (x2 - u)(x2 - u) + (x3 - u)(x3 - u) =
    
    x1^2 - 2ux1 + u^2 + x2^2 - 2ux2 + u^2 + x3^2 - 2ux3 + u^2 =
    
    (x1^2 + x2^2 + x3^2) + (u^2 + u^2 + u^2) + (- 2ux1 - 2ux2 - 2ux3) =
    (x1^2 + x2^2 + x3^2) + 3u^2 - ( 2ux1 + 2ux2 + 2ux3) =
    (x1^2 + x2^2 + x3^2) + 3u^2 - 2u( x1 + x2 + x3) =

    note: (x1 + x2 + x3) = 3 * u        <->        u = (x1 + x2 + x3) / 3
    
    (x1^2 + x2^2 + x3^2) + 3u^2 - 2u( x1 + x2 + x3) = (x1^2 + x2^2 + x3^2) + 3u^2 - 2u * 3u =
    (x1^2 + x2^2 + x3^2) + 3u^2 - 6u^2 = 
    (x1^2 + x2^2 + x3^2) - 3u^2 = 
    Sum(x**2) - N (Sum(x)/N)**2 = Sum(x**2) - N * (Sum(x)/N) * (Sum(x) / N) = Sum(x**2) - N (Sum(x)**2/N**2) = Sum(x**2) - Sum(x)**2 / N
    Sum(x**2) - 1/N * Sum(x)**2
    """
