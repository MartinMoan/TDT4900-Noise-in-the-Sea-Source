#!/usr/bin/env python3
import abc
from ast import arg
import enum
import os
import hashlib
import sys
import pathlib
from tabnanny import verbose
from typing import Iterable, Mapping, Type, Tuple
import pickle

import sklearn
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
from interfaces import ICustomDataset, IDatasetBalancer, ILogger, IAsyncWorker, IFolder, ILoggerFactory
from tracking.logger import Logger
from glider.audiodata import LabeledAudioData
from glider.clipping import CachedClippedDataset
from datasets.tensordataset import TensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from cacher import Cacher
from datasets.binjob import progress

class CachedDatasetBalancer(IDatasetBalancer):
    def __new__(cls, dataset: ICustomDataset, logger_factory: ILoggerFactory, worker: IAsyncWorker, force_recache=False, **kwargs) -> IDatasetBalancer:
        cacher = Cacher(logger_factory=logger_factory)
        init_args = (dataset, logger_factory, worker)
        hashable_arguments = {"dataset": dataset}
        return cacher.cache(DatasetBalancer, init_args=init_args, init_kwargs=kwargs, hashable_arguments=hashable_arguments, force_recache=force_recache)

class DatasetBalancer(IDatasetBalancer):
    def __init__(self, dataset: ICustomDataset, logger_factory: ILoggerFactory, worker: IAsyncWorker, verbose=True) -> None:
        super().__init__()
        self.logger = logger_factory.create_logger()
        self.worker = worker

        self._dataset = dataset
        self._split_labels: Mapping[str, Iterable[int]] = self._split_by_labels(dataset)
    
        if self._min_size == 0:
            raise Exception(f"Unable to balance dataset, because there is a label presence pair that has no values: {str({key: len(self._split_labels[key]) for key in self._split_labels.keys()})}")

        if verbose:
            self.log_balanced_stats()

    @property
    def _min_size(self):
        return np.min([len(self._split_labels[key]) for key in self._split_labels.keys()], axis=0)

    @property
    def _both_labels_indeces(self) -> Iterable[int]:
        return self._split_labels["both"]

    @property
    def _neither_labels_indeces(self) -> Iterable[int]:
        return self._split_labels["neither"]

    @property
    def _anthropogenic_indeces(self) -> Iterable[int]:
        return self._split_labels["anthropogenic"]

    @property
    def _biophonic_indeces(self) -> Iterable[int]:
        return self._split_labels["biophonic"]

    @property
    def _indeces_for_training(self) -> Mapping[str, Iterable[int]]:
        out = {}
        for key in self._split_labels.keys():
            indeces: Iterable[int] = np.array(self._split_labels[key])
            # train_part = np.random.choice(indeces, size=min_size, replace=False)
            # eval_part = indeces[~np.isin(indeces, train_part)]
            train_part = indeces[:self._min_size]
            
            out[key] = train_part
        return out
        
    @property
    def _indeces_for_eval(self) -> Mapping[str, Iterable[int]]:
        out = {}
        for key in self._split_labels.keys():
            indeces = np.array(self._split_labels[key])
            eval_part = indeces[self._min_size:]
            out[key] = eval_part
        return out
    
    def log_balanced_stats(self):
        self.logger.log(f"{self.__class__.__name__} class/label-presence distribution:")
        self.logger.log(f"Size of dataset: {len(self._dataset)}")
        self.logger.log("The instances under for training and eval will be split according to the current fold train/test split. And the eval only instances will be added to the test part of the split for every fold.")
        self.logger.log("---- FOR TRAINING AND EVAL ---- ")
        for key in self._indeces_for_training.keys():
            self.logger.log(f"Number of instances with label '{key}': {len(self._indeces_for_training[key])}")
        
        self.logger.log("---- FOR EVAL ONLY ---- ")
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

    def _func(self, dataset: ICustomDataset, start: int, stop: int):
        proc = multiprocessing.current_process()
        both = []
        anthropogenic = []
        biophonic = []
        neither = []
        for i in range(start, min(len(dataset), stop)):
            should_log, percentage = progress(i, start, stop)
            if should_log:
                self.logger.log(f"BalancingWorker PID {proc.pid} - {percentage:.2f}%")

            audiodata: LabeledAudioData = dataset[i]
            class_presence = audiodata.labels.source_class.unique()

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

    def _agg(self, parts: Iterable[Mapping[str, Iterable]]) -> Mapping[str, Iterable]:
        output = {}
        for mapping in parts:
            for key, value in mapping.items():
                if key in output.keys():
                    output[key] += value
                else:
                    output[key] = value
        return output

    def _split_by_labels(self, dataset: ICustomDataset) -> Mapping[str, Iterable[int]]:
        return self.worker.apply(dataset, self._func, aggregation_method=self._agg)
        
    def eval_only_indeces(self) -> Iterable[int]:
        indeces = np.concatenate([self._indeces_for_eval[key] for key in self._indeces_for_eval], axis=0)
        np.random.shuffle(indeces)
        return indeces.astype(int)

    def train_indeces(self) -> Iterable[int]:
        indeces = np.concatenate([self._indeces_for_training[key] for key in self._indeces_for_training], axis=0)
        np.random.shuffle(indeces)
        return indeces.astype(int)

class BalancedKFolder(IFolder):
    def __init__(
        self, 
        n_splits=5, 
        *, 
        shuffle=False, 
        random_state=None, 
        balancer_ref: Type[IDatasetBalancer] = DatasetBalancer,
        balancer_args: Tuple[any, ...] = (),
        balancer_kwargs: Mapping[str, any] = {}):

        super().__init__()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._balancer_ref = balancer_ref
        self._balancer_args = balancer_args
        self._balancer_kwargs = balancer_kwargs

    def split(self, filelength_dataset: TensorAudioDataset):
        if filelength_dataset is None:
            raise ValueError("Input dataset is None")
        
        if not isinstance(filelength_dataset, TensorAudioDataset):
            raise TypeError
        
        if filelength_dataset._dataset is None:
            raise ValueError

        if not isinstance(filelength_dataset._dataset, ICustomDataset):
            raise TypeError
        
        balancer = self._balancer_ref(filelength_dataset._dataset, *self._balancer_args, **self._balancer_kwargs)

        all_training_indeces = balancer.train_indeces()
        eval_only_indeces = balancer.eval_only_indeces()
        
        error_indeces = [idx for idx in eval_only_indeces if idx in all_training_indeces]
        if len(error_indeces) != 0:
            raise Exception(f"There are indeces that should only be used for eval that are also present in the training indeces.")
        
        folder = sklearn.model_selection.KFold(self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for (train, eval_indeces) in folder.split(all_training_indeces):
            all_eval_indeces = np.concatenate([all_training_indeces[eval_indeces], eval_only_indeces], axis=0, dtype=int)
            yield (all_training_indeces[train], all_eval_indeces)

    @property
    def properties(self) -> Mapping[str, any]:
        return {"k_folds": self.n_splits, "shuffle": self.shuffle, "random_state": self.random_state}

class BalancedDatasetDecorator(ICustomDataset):
    def __init__(self, dataset: ICustomDataset, balancer: IDatasetBalancer, force_recarche=False, **kwargs) -> None:
        super().__init__()
        self._dataset = dataset
        self._indeces = balancer.train_indeces()

    def __len__(self) -> int:
        return len(self._indeces)
    
    def __getitem__(self, index: int) -> LabeledAudioData:
        return self._dataset[self._indeces[index]]
    
    def classes(self) -> Mapping[str, int]:
        return self._dataset.classes()

    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        return self._dataset.example_shapes()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}( _dataset: {repr(self._dataset)}, _indeces: {self._indeces} )"

if __name__ == "__main__":
    clip_duration_seconds = 10.0
    clip_overlap_seconds = 2.0
    dataset = CachedClippedDataset(clip_duration_seconds=clip_duration_seconds, clip_overlap_seconds=clip_overlap_seconds)
    
    d = TensorAudioDataset(
        dataset, 
        BinaryLabelAccessor(),
        MelSpectrogramFeatureAccessor()
    )

    from datasets.binjob import Binworker, progress

    folder = BalancedKFolder(n_splits=5, shuffle=False, balancer_ref=CachedDatasetBalancer)
    for fold, (train_indeces, test_indeces) in enumerate(folder.split(d)):
        for idx in train_indeces:
            index, X, Y = d[idx]
            print(Y)