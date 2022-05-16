#!/usr/bin/env python3 
import os
import argparse
import sys
import pathlib
from typing import Tuple, Iterable, Union, List
import hashlib
import pathlib
import pickle

import git
import torch
from rich import print
import wandb
import numpy as np
import torchmetrics

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from interfaces import ILoggerFactory, ITensorAudioDataset, IAsyncWorker, IDatasetBalancer
from tracking.logger import SlurmLogger
from tracking.loggerfactory import LoggerFactory

from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import DatasetBalancer, CachedDatasetBalancer
from datasets.binjob import Binworker
from datasets.tensordataset import BinaryLabelAccessor, MelSpectrogramFeatureAccessor, TensorAudioDataset

def hash(*args, **kwargs):
    hasher = hashlib.sha256()
    for value in args:
        hasher.update(repr(value).encode())
    for key, value in kwargs.items():
        hasher.update(repr(f"{key}=").encode())
        hasher.update(repr(value).encode())
    hex_hash = hasher.hexdigest()
    return hex_hash

def get_clips_cache_dir():
    return config.CACHE_DIR.joinpath("clipping")

def get_balancer_cache_dir():
    return config.CACHE_DIR.joinpath("balancing")

def get_pickled_clips():
    return [glob for glob in get_clips_cache_dir().glob("**/*.pickle")]

def get_pickled_balancers():
    return [glob for glob in get_balancer_cache_dir().glob("**/*.pickle")]

def get_clips(
    logger_factory: ILoggerFactory, 
    worker: IAsyncWorker, 
    clip_duration_seconds: float = None, 
    clip_overlap_seconds: float = None, 
    clip_nsamples: int = None, 
    overlap_nsamples: int = None
    ) -> ClippedDataset:
    
    logger = logger_factory.create_logger()

    hashable_arguments = dict(
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds,
        clip_nsamples=clip_nsamples,
        overlap_nsamples=overlap_nsamples,
    )
    
    hex_hash = hash(**hashable_arguments)
    expected_pickle_path = get_clips_cache_dir().joinpath(f"{hex_hash}.pickle")

    slurm_procid = int(os.environ.get("SLURM_PROCID", default=-1))
    if slurm_procid == 0 or slurm_procid == -1:
        # slurm_procid == 0: Running as slurm managed process, and has global rank 0, in which case we should instantiate the dataset as a new object.
        # slurm_procid == -1: Either running on local dev machine, or running on cluster as standalone process not managed by slurm.
        clips = ClippedDataset(
            logger_factory=logger_factory,
            worker=worker,
            clip_overlap_seconds=clip_overlap_seconds,
            clip_duration_seconds=clip_duration_seconds,
            clip_nsamples=clip_nsamples,
            overlap_nsamples=overlap_nsamples
        )

        if not expected_pickle_path.parent.exists():
            expected_pickle_path.parent.mkdir(parents=True, exist_ok=False)

        logger.log(f"Pickling ClippedDataset object to {expected_pickle_path}")
        with open(expected_pickle_path, "wb") as binary_file:
            pickle.dump(clips, binary_file)
        
        return clips
    
    logger.log(f"Pickling ClippedDataset object from {expected_pickle_path}")
    with open(expected_pickle_path, "rb") as binary_file:
        return pickle.load(binary_file)

def get_balancer(
    clips: ClippedDataset, 
    logger_factory: ILoggerFactory,
    worker: IAsyncWorker) -> IDatasetBalancer:

    logger = logger_factory.create_logger()

    hex_hash = hash(clips=clips)
    expected_pickle_path = get_balancer_cache_dir().joinpath(f"{hex_hash}.pickle")
    
    slurm_procid = int(os.environ.get("SLURM_PROCID", default=-1))
    if slurm_procid == 0 or slurm_procid == -1:
        # slurm_procid == 0: Running as slurm managed process, and has global rank 0, in which case we should instantiate the dataset as a new object.
        # slurm_procid == -1: Either running on local dev machine, or running on cluster as standalone process not managed by slurm.
        
        balancer = DatasetBalancer(
            dataset=clips, 
            logger_factory=logger_factory, 
            worker=worker,
            verbose=True
        )

        if not expected_pickle_path.parent.exists():
            expected_pickle_path.parent.mkdir(parents=True, exist_ok=False)
        
        logger.log(f"Pickling DatasetBalancer object to {expected_pickle_path}")
        with open(expected_pickle_path, "wb") as binary_file:
            pickle.dump(balancer, binary_file)
        
        return balancer

    logger.log(f"Pickling DatasetBalancer object from {expected_pickle_path}")
    with open(expected_pickle_path, "rb") as binary_file:
        return pickle.load(binary_file)

def create_tensorset(
    logger_factory: ILoggerFactory, 
    nfft: int, 
    nmels: int, 
    hop_length: int, 
    clip_duration_seconds: float, 
    clip_overlap_seconds: float) -> Tuple[ITensorAudioDataset, Iterable[int], Iterable[int]]:

    clips = get_clips(
        logger_factory=logger_factory,
        worker=Binworker(),
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds
    )

    balancer = get_balancer(
        clips=clips,
        logger_factory=logger_factory,
        worker=Binworker()
    )
    balancer.log_balanced_stats()

    label_accessor = BinaryLabelAccessor()
    feature_accessor = MelSpectrogramFeatureAccessor(
        logger_factory=logger_factory,
        n_mels=nmels,
        n_fft=nfft,
        hop_length=hop_length,
        scale_melbands=False,
        verbose=True
    )

    tensorset = TensorAudioDataset(
        dataset=clips,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor,
        logger_factory=logger_factory
    )

    return tensorset, balancer

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nmels", type=int, required=True)
    parser.add_argument("-nfft", type=int, required=True)
    parser.add_argument("-hop_length", type=int, required=True)
    parser.add_argument("-clip_duration_seconds", type=float, required=True)
    parser.add_argument("-clip_overlap_seconds", type=float, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = getargs()
    proc = os.environ.get("SLURM_PROCID")
    print(f"SLURM_PROCID: {proc}")

    logger_factory = LoggerFactory(logger_type=SlurmLogger)

    tensorset, balancer = create_tensorset(
        logger_factory=logger_factory,
        nfft=args.nfft,
        nmels=args.nmels,
        hop_length=args.hop_length,
        clip_duration_seconds=args.clip_duration_seconds,
        clip_overlap_seconds=args.clip_overlap_seconds
    )

# python initdata.py -nmels 128 -hop_length 512 -nfft 1024 -clip_duration_seconds 10.0 -clip_overlap_seconds 4.0