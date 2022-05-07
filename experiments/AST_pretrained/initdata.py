#!/usr/bin/env python3 
import argparse
import sys
import pathlib
from typing import Tuple, Iterable

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from interfaces import ILoggerFactory, ITensorAudioDataset
from tracking.logger import Logger, LogFormatter
from tracking.loggerfactory import LoggerFactory

from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import DatasetBalancer, CachedDatasetBalancer
from datasets.binjob import Binworker
from datasets.tensordataset import BinaryLabelAccessor, MelSpectrogramFeatureAccessor, TensorAudioDataset

def create_tensorset(
    logger_factory: ILoggerFactory, 
    nfft: int, 
    nmels: int, 
    hop_length: int, 
    clip_duration_seconds: float, 
    clip_overlap_seconds: float, 
    force_recache: bool = False) -> Tuple[ITensorAudioDataset, Iterable[int], Iterable[int]]:

    clips = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=Binworker(),
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds,
        force_recache=force_recache
    )

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

    balancer = CachedDatasetBalancer(
        dataset=clips,
        logger_factory=logger_factory,
        worker=Binworker(),
        verbose=True,
        force_recache=force_recache
    )

    eval_only_indeces = balancer.eval_only_indeces()
    train_indeces = balancer.train_indeces()

    return tensorset, eval_only_indeces, train_indeces

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
    logger_factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(),
        logger_kwargs=dict(logformatter=LogFormatter())
    )

    create_tensorset(
        logger_factory=logger_factory,
        nfft=args.nfft,
        nmels=args.nmels,
        hop_length=args.hop_length,
        clip_duration_seconds=args.clip_duration_seconds,
        clip_overlap_seconds=args.clip_overlap_seconds,
        force_recache=True
    )
