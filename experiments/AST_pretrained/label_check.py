#!/usr/bin/env python3

import argparse
import pathlib
import sys
import multiprocessing

import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler, SequentialSampler
from rich import print
import git
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from GLIDER import GLIDER
from clipping import ClippedDataset, ClippingCacheDecorator
from ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
import trainer
from ITensorAudioDataset import FileLengthTensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor, ITensorAudioDataset
from IMetricComputer import BinaryMetricComputer
from IDatasetBalancer import BalancedKFolder, DatasetBalancer, CachedDatasetBalancer
from ASTWrapper import ASTWrapper
from limiting import DatasetLimiter

def main():
    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    nmels = 128
    hop_length = 512

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    clip_dataset = ClippingCacheDecorator(
        clip_nsamples = clip_length_samples, 
        overlap_nsamples = clip_overlap_samples
    )

    dataset = FileLengthTensorAudioDataset(
        dataset = clip_dataset,
        label_accessor=BinaryLabelAccessor(),
        feature_accessor=MelSpectrogramFeatureAccessor(n_mels=nmels)
    )

    folder = BalancedKFolder(n_splits=5, shuffle=False)

    for fold, (training_samples, test_samples) in enumerate(folder.split(dataset)):
        print(f"Fold: {fold}")
        # sampler = SequentialSampler(training_samples)
        # trainset = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=multiprocessing.cpu_count())
        # for batch, (index, X, Y) in enumerate(trainset):
        #     print(batch, torch.unique(Y))

        def cat(t1, t2):
            if t1 is None:
                return t2
            return torch.cat((t1, t2))
        
        testset = DataLoader(
            dataset, 
            batch_size=16, 
            sampler=SubsetRandomSampler(test_samples), 
            num_workers=multiprocessing.cpu_count()
        )
        
        all_ys = None
        for i, (index, X, Y) in enumerate(testset):
            print(X, X.shape, Y, Y.shape)
            all_ys = cat(all_ys, Y)
        

if __name__ == "__main__":
    main()