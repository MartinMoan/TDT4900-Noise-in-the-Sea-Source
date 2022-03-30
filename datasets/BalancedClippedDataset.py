#!/usr/bin/env python3
import abc
from typing import Iterable, Mapping, Tuple
import sys
import torch
from ICustomDataset import ICustomDataset
from glider.audiodata import LabeledAudioData
from glider.clipping import ClippedDataset
from ITensorAudioDataset import ITensorAudioDataset, FileLengthTensorAudioDataset, ILabelAccessor, IFeatureAccessor, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
import multiprocessing
from multiprocessing import Pool
import math
import numpy as np

class BalancedDataset(FileLengthTensorAudioDataset):
    def __init__(self, dataset: ICustomDataset, label_accessor: ILabelAccessor, feature_accessor: IFeatureAccessor) -> None:
        super().__init__(dataset, label_accessor, feature_accessor)
        self._dataset = dataset
        _labeled, _unlabeled = self._preprocess(dataset)
        self._labeled_indeces = _labeled
        self._unlabeled_indeces = _unlabeled

        
        _percentage_labeled = len(self._labeled_indeces) / len(self._dataset) # The percentage of clips/files that are labeled (are biophonic and/or anthropogenic)
        _num_unlabeled_to_use = int(len(self._unlabeled_indeces) * _percentage_labeled) # Use an equal percentage of unlabeled files, and treat them as if they were labeled as not-biophonic and not-anthropogenic (e.g.: [0, 0])

        self._unlabeled_indeces_to_use = np.random.choice(self._unlabeled_indeces, size=_num_unlabeled_to_use, replace=False) # select a random subset of the unlabeled clips/files

        self._indeces_to_use = np.concatenate((self._labeled_indeces, self._unlabeled_indeces_to_use), axis=0)

        self._label_accessor = label_accessor
        self._feature_accessor = feature_accessor

    def _get_labeled_unlabeled(self, dataset: ICustomDataset, start: int, end: int) -> Tuple[Iterable[int], Iterable[int]]:
        proc = multiprocessing.current_process()

        labeled = []
        unlabeled = []
        for i in range(start, min(end, len(dataset))):
            part = math.ceil((end - start) * 0.025)
            if (i - start) % part == 0:
                percentage = ((i - start) / (end - start)) * 100
                print(f"Balancing: {proc.name} {percentage:.2f}%")
                
            labeled_audio_data: LabeledAudioData = dataset[i]
            if len(labeled_audio_data.labels) > 0:
                labeled.append(i)
            else:
                unlabeled.append(i)
        return labeled, unlabeled

    def _preprocess(self, dataset: ICustomDataset) -> Tuple[Iterable[int], Iterable[int]]:
        """Finds all the indeces of ICustomDataset argument that has active labels

        Returns:
            Tuple[Iterable[int], Iterable[int]]: A tuple of two iterables of indeces to the ICustomDataset argument. The first iterable contains the indeces of labeled instances, the second iterable containing the unlabeled instance indeces. 
        """
        
        n_processes = multiprocessing.cpu_count()
        with Pool(processes=n_processes) as pool:
            bin_size = math.ceil(len(dataset) / n_processes)
            
            bins = [(start, (start + bin_size)) for start in range(0, len(dataset), bin_size)]
            
            tasks = []
            for start, end in bins:
                task = pool.apply_async(self._get_labeled_unlabeled, (dataset, start, end))
                tasks.append(task)

            labeled, unlabeled = [], []
            for task in tasks:
                labeled_subset, unlabeled_subset = task.get()
                labeled += labeled_subset
                unlabeled += unlabeled_subset
            return labeled, unlabeled

    def __len__(self) -> int:
        return len(self._indeces_to_use)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Returns a tuple (index: int, X: torch.Tensor, Y: torch.Tensor)"""
        return super().__getitem__(self._indeces_to_use[index])

if __name__ == "__main__":
    dataset = ClippedDataset(clip_duration_seconds=10.0, clip_overlap_seconds=2.0)
    balanced_dataset = BalancedDataset(dataset, feature_accessor=MelSpectrogramFeatureAccessor(), label_accessor=BinaryLabelAccessor())
    for i in range(min(len(balanced_dataset), 10)):
        print(balanced_dataset[i])