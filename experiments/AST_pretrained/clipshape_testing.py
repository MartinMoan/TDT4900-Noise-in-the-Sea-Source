#!/usr/bin/env python3
import multiprocessing
import multiprocessing.pool
import sys
import pathlib

import git
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from tracking.logger import Logger, LogFormatter
from tracking.loggerfactory import LoggerFactory

from datasets.balancing import CachedDatasetBalancer
from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import BalancedKFolder
from datasets.tensordataset import TensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from datasets.binjob import Binworker

import config

def init(
        logger_factory,
        n_mels,
        n_fft,
        hop_length,
        clip_length_samples=None,
        clip_overlap_samples=None,
        clip_duration_seconds=None,
        clip_overlap_seconds=None
    ):

    worker = Binworker(
        pool_ref=multiprocessing.pool.Pool,
        n_processes=multiprocessing.cpu_count(),
        timeout_seconds=None
    )

    clipped_dataset = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=worker,
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds,
        clip_nsamples=clip_length_samples,
        overlap_nsamples=clip_overlap_samples,
    )

    return clipped_dataset

def createspect(logger_factory, clip, n_mels, n_fft, hop_length, storepath):
    print(n_mels, n_fft, hop_length)
    feature_accessor = MelSpectrogramFeatureAccessor(
        logger_factory=logger_factory, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        scale_melbands=False,
        verbose=True
    )
    X = feature_accessor(clip)

    plt.imshow(X.detach().squeeze().numpy(), aspect="auto")
    plt.suptitle(clip.filepath.name)
    plt.title(", ".join(clip.labels.source_class_specific.unique()))
    filename=f"n_mels_{n_mels}_hop_length_{hop_length}_n_fft_{n_fft}.png"
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig(storepath.joinpath(filename), dpi=500)

def main():
    logger_factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(),
        logger_kwargs=dict(logformatter=LogFormatter(),)
    )

    n_mels = 1024 # number of mel frequency bands
    n_fft = 8184*2 # length of window to compute Fourier Transform over. Higher - create frequency resolution, lower - greater time resolution
    hop_length = 512 # length between successive Fourier Transform windows

    mels = [2**i for i in range(6, 14)]
    ffts = [2**i for i in range(6, 16)]
    hop_lengths = [2**i for i in range(6, 14)]
    print("mels:", mels)
    print("ffts:", ffts)
    print("hop_lengths:", hop_lengths)

    clip_duration_seconds = 30.0
    clip_overlap_seconds = 10.0

    dataset = init(
        logger_factory,
        n_mels,
        n_fft,
        hop_length,
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds
    )

    example_index = 15375
    path = config.HOME_PROJECT_DIR.joinpath("spectparams_examples")
    if not path.exists():
        path.mkdir(parents=False, exist_ok=False)

    clip = dataset[example_index]

    import multiprocessing
    from multiprocessing.pool import Pool

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = []
        for n_mels in mels:
            for n_fft in ffts:
                for hop_length in hop_lengths:
                    args = (logger_factory, clip, n_mels, n_fft, hop_length, path)
                    task = pool.apply_async(createspect, args, kwds={})
                    tasks.append(task)
        for task in tasks:
            task.get()

if __name__ == "__main__":
    main()
