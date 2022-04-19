#!/usr/bin/env python3
import abc
from ast import arg
import enum
import os
import hashlib
import sys
import pathlib
from tabnanny import verbose
from typing import Iterable, Mapping
import pickle

import sklearn
from ICustomDataset import ICustomDataset
from glider.audiodata import LabeledAudioData
from glider.clipping import ClippedDataset, ClippingCacheDecorator
from ITensorAudioDataset import ITensorAudioDataset, FileLengthTensorAudioDataset, ILabelAccessor, IFeatureAccessor, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from cacher import Cacher
import multiprocessing
from multiprocessing import Pool
import math
import numpy as np
from rich import print
import sklearn
import sklearn.model_selection
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from logger import Logger, ILogger

class IDatasetBalancer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eval_only_indeces() -> Iterable[int]:
        """Set the dataset to eval mode, the dataset ouput will not be balanced."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_indeces() -> Iterable[int]:
        raise NotImplementedError

BALANCER_CACHE_DIR = config.CACHE_DIR.joinpath("balancing")
if not BALANCER_CACHE_DIR.exists():
    BALANCER_CACHE_DIR.mkdir(parents=True, exist_ok=False)

class BalancerCacheDecorator(IDatasetBalancer):
    def __new__(cls, dataset: ICustomDataset, force_recache=False, **kwargs) -> IDatasetBalancer:
        hashable_arguments = repr(dataset).encode()
        args = (dataset,)
        cacher = Cacher()
        return cacher.cache(BALANCER_CACHE_DIR, DatasetBalancer, *args, hashable_arguments=hashable_arguments, force_recache=force_recache, **kwargs)

class DatasetBalancer(IDatasetBalancer):
    """A ITensorAudioDataset implentation that ensures that the number of unlabeled instances to use for training is equal to the average size of each of the types of labeled instances. 

    Because any ICustomDataset can contain instances with any ONE of the following label presence pairs:
        [
            ("Anthropogenic", "Biogenic"), # E.g. 'both'
            ("Not anthropogenic", "Biogenic"),
            ("Anthropogenic", "Not Biogenic"), 
            ("Not anthropogenic", "Not Biogenic"), # E.g. 'neither'
        ]
    All but the last (neither) represent instances that are manually labeled by a human or an automatic detector. Whereas the 'neither' label instance pair represents instances that are not manually marked with any sound events. The lack of such manual labeling for these instances can have one of the two following causes:
        1. The audio was inspected, but did not contain any sound events of anthropogenic not biogenic origin, and therefore no labels was registered.
        2. The audio was never inspecte, resulting in no labels being registered for the audio. 
    
    In either case how to interpret the 'neither' class is important to ensure that the performance of models trained using the data is not affected by label/class imbalance. Because the 'neither' class is dominant within the GLIDER dataset.

    However, the instance balancing should only be used during training, and not during evaluation. 
    

    Args:
        FileLengthTensorAudioDataset (_type_): _description_
    """
    def __init__(self, dataset: ICustomDataset, verbose=True, logger: ILogger = Logger()) -> None:
        super().__init__()
        self.logger = logger
        
        self._dataset = dataset
        self._split_labels: Mapping[str, Iterable[int]] = self._split_by_labels(dataset)

        self._both_labels_indeces: Iterable[int] = self._split_labels["both"]
        self._neither_labels_indeces: Iterable[int] = self._split_labels["neither"]
        self._anthropogenic_indeces: Iterable[int] = self._split_labels["anthropogenic"]
        self._biophonic_indeces: Iterable[int] = self._split_labels["biophonic"]

        average_num_examples_per_label: float = np.mean([len(self._both_labels_indeces), len(self._anthropogenic_indeces), len(self._biophonic_indeces)])
        min_size = np.min([len(self._split_labels[key]) for key in self._split_labels.keys()], axis=0)
        if min_size == 0:
            raise Exception(f"Unable to balance dataset, because there is a label presence pair that has no values: {str({key: len(self._split_labels[key]) for key in self._split_labels.keys()})}")

        self._indeces_for_training = {}
        self._indeces_for_eval = {}
        for key in self._split_labels.keys():
            indeces: Iterable[int] = np.array(self._split_labels[key])
            train_part = np.random.choice(indeces, size=min_size, replace=False)
            eval_part = indeces[~np.isin(indeces, train_part)]
            self._indeces_for_training[key] = train_part
            self._indeces_for_eval[key] = eval_part
        
        if verbose:
            self.logger.log(f"{self.__class__.__name__} class/label-presence distribution:")
            self.logger.log(f"Size of dataset: {len(self._dataset)}")
            self.logger.log("The instances under for training and eval will be split according to the current fold train/test split. And the eval only instances will be added to the test part of the split for every fold.")
            self.logger.log("---- FOR TRAINING AND EVAL ---- ")
            for key in self._indeces_for_training.keys():
                self.logger.log(f"Number of instances with label '{key}': {len(self._indeces_for_training[key])}")
            
            self.logger.log("---- FOR EVAL ONLY ---- ")
            self.logger.log("")
            for key in self._indeces_for_eval.keys():
                self.logger.log(f"Number of instances with label '{key}': {len(self._indeces_for_eval[key])}")
            self.logger.log("---- ORIGINAL DISTRIBUTION BEFORE BALANCING ----")
            for key in self._split_labels.keys():
                self.logger.log(f"Number of instances with label '{key}': {len(self._split_labels[key])}")

            num_for_training = np.sum([len(self._indeces_for_training[key]) for key in self._indeces_for_training.keys()])
            num_for_eval = np.sum([len(self._indeces_for_eval[key]) for key in self._indeces_for_eval.keys()])
            self.logger.log(f"Total number of instances for training: {num_for_training}")
            self.logger.log(f"Total number of instances for eval: {num_for_eval}")
            self.logger.log(f"Total number of instances for both training and eval: {(num_for_training + num_for_eval)}")
            self.logger.log(f"Input dataset length: {len(self._dataset)}")
            self.logger.log(f"The number of instances is as expected?: {(num_for_training + num_for_eval) == len(self._dataset)}")

    def _split_by_labels_poolfunc(self, dataset: ICustomDataset, start: int, end: int) -> Mapping[str, Iterable[int]]:
        proc = multiprocessing.current_process()

        both = []
        anthropogenic = []
        biophonic = []
        neither = []
        
        for i in range(start, min(end, len(dataset))):
            part = math.ceil((end - start) * 0.05)
            if (i - start) % part == 0:
                percentage = ((i - start) / (end - start)) * 100
                self.logger.log(f"BalancingWorker PID {proc.pid} - {percentage:.2f}%")
                
            labeled_audio_data: LabeledAudioData = dataset[i]
            class_presence = labeled_audio_data.labels.source_class.unique()
            if len(class_presence) == 0:
                neither.append(i)
            elif len(class_presence) == 1:
                if class_presence[0] == "Biophonic":
                    biophonic.append(i)
                elif class_presence[0] == "Anthropogenic":
                    anthropogenic.append(i)
            elif len(class_presence) == 2:
                both.append(i)
            else:
                raise Exception(f"Received an unexpected number of unique source_class values in the labeled_audio_data.labels pd.DataFrame. Expected 0, 1 or 2 unique values but received {len(class_presence)} with the following values: {class_presence}")

        return {"both": both, "anthropogenic": anthropogenic, "biophonic": biophonic, "neither": neither}

    def _split_by_labels(self, dataset: ICustomDataset) -> Mapping[str, Iterable[int]]:
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
                task = pool.apply_async(self._split_by_labels_poolfunc, (dataset, start, end))
                tasks.append(task)

            output = {}
            for task in tasks:
                subsets = task.get()
                for key in subsets.keys():
                    if key not in output.keys():
                        output[key] = subsets[key]
                    else:
                        output[key] += subsets[key]
            return output

    def eval_only_indeces(self) -> Iterable[int]:
        indeces = np.concatenate([self._indeces_for_eval[key] for key in self._indeces_for_eval], axis=0)
        np.random.shuffle(indeces)
        return indeces.astype(int)

    def train_indeces(self) -> Iterable[int]:
        indeces = np.concatenate([self._indeces_for_training[key] for key in self._indeces_for_training], axis=0)
        np.random.shuffle(indeces)
        return indeces.astype(int)

class BalancedKFolder(sklearn.model_selection.KFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, filelength_dataset: FileLengthTensorAudioDataset):
        if filelength_dataset is None:
            raise ValueError("Input dataset is None")
        
        if not isinstance(filelength_dataset, FileLengthTensorAudioDataset):
            raise TypeError
        
        if filelength_dataset._dataset is None:
            raise ValueError

        if not isinstance(filelength_dataset._dataset, ICustomDataset):
            raise TypeError
        
        balancer = BalancerCacheDecorator(filelength_dataset._dataset)

        all_training_indeces = balancer.train_indeces()
        eval_only_indeces = balancer.eval_only_indeces()
        
        error_indeces = [idx for idx in eval_only_indeces if idx in all_training_indeces]
        if len(error_indeces) != 0:
            raise Exception(f"There are indeces that should only be used for eval that are also present in the training indeces.")
        
        for (train, eval_indeces) in super().split(all_training_indeces):
            # eval_indeces = np.concatenate([all_training_indeces[eval], eval_only_indeces], axis=0, dtype=int)
            yield (all_training_indeces[train], eval_indeces)

class BalancedDatasetDecorator(ICustomDataset):
    def __init__(self, dataset: ICustomDataset, force_recarche=False, **kwargs) -> None:
        super().__init__()
        self._dataset = dataset
        balancer = BalancerCacheDecorator(dataset, force_recache=force_recarche, **kwargs)
        self._indeces = balancer.train_indeces()

    def __len__(self) -> int:
        return len(self._indeces)
    
    def __getitem__(self, index: int) -> LabeledAudioData:
        return self._dataset[self._indeces[index]]
    
    def classes(self) -> Mapping[str, int]:
        return self._dataset.classes()

    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        return self._dataset.example_shapes()

if __name__ == "__main__":
    clip_duration_seconds = 10.0
    clip_overlap_seconds = 2.0
    dataset = ClippingCacheDecorator(clip_duration_seconds=clip_duration_seconds, clip_overlap_seconds=clip_overlap_seconds)
    
    d = FileLengthTensorAudioDataset(
        dataset, 
        BinaryLabelAccessor(),
        MelSpectrogramFeatureAccessor()
    )

    folder = BalancedKFolder(n_splits=5, shuffle=False)
    import torch
    for fold, (train_indeces, test_indeces) in enumerate(folder.split(d)):
        for idx in test_indeces:
            index, X, Y = d[idx]
            print(Y)