#!/usr/bin/env python3 
import os
import sys
import pathlib
from typing import Tuple, Iterable
import hashlib
import pathlib
import pickle

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from interfaces import ILoggerFactory, ITensorAudioDataset, IAsyncWorker, IDatasetBalancer

from datasets.glider.clipping import ClippedDataset
from datasets.balancing import DatasetBalancer
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
    clip_duration_seconds: float = None, 
    clip_overlap_seconds: float = None, 
    clip_nsamples: int = None, 
    overlap_nsamples: int = None
    ) -> ClippedDataset:

    hashable_arguments = dict(
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds,
        clip_nsamples=clip_nsamples,
        overlap_nsamples=overlap_nsamples,
    )
    
    hex_hash = hash(**hashable_arguments)
    expected_pickle_path = get_clips_cache_dir().joinpath(f"{hex_hash}.pickle")

    slurm_procid = int(os.environ.get("SLURM_PROCID", default=-1))
    if (slurm_procid == 0 or slurm_procid == -1) and not expected_pickle_path.exists():
        # slurm_procid == 0: Running as slurm managed process, and has global rank 0, in which case we should instantiate the dataset as a new object.
        # slurm_procid == -1: Either running on local dev machine, or running on cluster as standalone process not managed by slurm.
        clips = ClippedDataset(
            clip_overlap_seconds=clip_overlap_seconds,
            clip_duration_seconds=clip_duration_seconds,
            clip_nsamples=clip_nsamples,
            overlap_nsamples=overlap_nsamples
        )

        if not expected_pickle_path.parent.exists():
            expected_pickle_path.parent.mkdir(parents=True, exist_ok=False)

        print(f"Pickling ClippedDataset object to {expected_pickle_path}")
        with open(expected_pickle_path, "wb") as binary_file:
            pickle.dump(clips, binary_file)
        
        return clips
    
    print(f"Pickling ClippedDataset object from {expected_pickle_path}")
    with open(expected_pickle_path, "rb") as binary_file:
        return pickle.load(binary_file)

def get_balancer(clips: ClippedDataset) -> IDatasetBalancer:
    hex_hash = hash(clips=clips)
    expected_pickle_path = get_balancer_cache_dir().joinpath(f"{hex_hash}.pickle")
    
    slurm_procid = int(os.environ.get("SLURM_PROCID", default=-1))
    if (slurm_procid == 0 or slurm_procid == -1) and not expected_pickle_path.exists():
        # slurm_procid == 0: Running as slurm managed process, and has global rank 0, in which case we should instantiate the dataset as a new object.
        # slurm_procid == -1: Either running on local dev machine, or running on cluster as standalone process not managed by slurm.
        
        balancer = DatasetBalancer(
            dataset=clips,
            verbose=True
        )

        if not expected_pickle_path.parent.exists():
            expected_pickle_path.parent.mkdir(parents=True, exist_ok=False)
        
        print(f"Pickling DatasetBalancer object to {expected_pickle_path}")
        with open(expected_pickle_path, "wb") as binary_file:
            pickle.dump(balancer, binary_file)
        
        return balancer

    print(f"Pickling DatasetBalancer object from {expected_pickle_path}")
    with open(expected_pickle_path, "rb") as binary_file:
        return pickle.load(binary_file)

def create_tensorset(
    nfft: int, 
    nmels: int, 
    hop_length: int, 
    clip_duration_seconds: float, 
    clip_overlap_seconds: float) -> Tuple[ITensorAudioDataset, Iterable[int], Iterable[int]]:

    clips = get_clips(
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds
    )

    balancer = get_balancer(clips=clips)

    label_accessor = BinaryLabelAccessor()
    feature_accessor = MelSpectrogramFeatureAccessor(
        n_mels=nmels,
        n_fft=nfft,
        hop_length=hop_length,
        scale_melbands=False,
        verbose=True
    )

    tensorset = TensorAudioDataset(
        dataset=clips,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor
    )

    return tensorset, balancer