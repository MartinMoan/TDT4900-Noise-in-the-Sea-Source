#!/usr/bin/env python3
from datetime import datetime, timedelta
import multiprocessing
import pathlib
import sys
from typing import Tuple, Mapping, Iterable, Union, Set
from multiprocessing import Pool
import math

import git
import numpy as np
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from interfaces import ITensorAudioDataset, ICustomDataset, IDatasetVerifier, ILogger, ILoggerFactory, IAsyncWorker
from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.glider.audiodata import LabeledAudioData
from datasets.tensordataset import TensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from datasets.limiting import DatasetLimiter
from datasets.balancing import DatasetBalancer, CachedDatasetBalancer, BalancedDatasetDecorator
from tracking.logger import Logger
from datasets.binjob import progress, Binworker

class ValueCount:
    def __init__(self, values: Iterable[Union[float, int]], count: int = 0):
        self.values = values
        self.count = count
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ValueCount):
            return other.values == self.values
        return False
    
    def __str__(self) -> str:
        obj = {
            "values": self.values,
            "count": self.count
        }
        return str(obj)
    
    def __repr__(self) -> str:
        return self.__str__()

class BinaryTensorDatasetVerifier(IDatasetVerifier):
    def __init__(self, logger_factory: ILoggerFactory, worker: IAsyncWorker, verbose: bool = True) -> None:
        self._verbose = verbose
        self.logger = logger_factory.create_logger()
        self.worker = worker
            
    def verify(self, dataset: ITensorAudioDataset) -> bool:
        self.logger.log("Verifying dataset...")
        unique_labels = [('Anthropogenic', 'Biophonic'), ('Anthropogenic',), ('Biophonic',), ()]
        found_labels = []
        found_indeces = []
        shuffled_indeces = np.random.choice(np.arange(len(dataset)), len(dataset), replace=False)
        for i in shuffled_indeces:
            audiodata = dataset.audiodata(i)

            labels = tuple(np.sort(audiodata.labels.source_class.unique(), axis=0))
            
            if labels not in found_labels:
                found_labels.append(labels)
                found_indeces.append(i)
            
            if set(found_labels) == set(unique_labels):
                u = [[0, 0], [0, 1], [1, 0], [1, 1]]
                f = []
                uqx = set([])
                for j in found_indeces:
                    X, Y = dataset[j]
                    lY = list(Y.numpy().astype(int))
                    if lY in u:
                        f.append(lY)

                    uqx = uqx.union(set(list(np.unique(X.numpy()))))
                    
                    if len(u) == len(f):
                        u = list(np.sort(u, axis=0))
                        f = list(np.sort(f, axis=0))
                        a = True
                        for k in range(len(u)):
                            for e in range(len(u[k])):
                                if u[k][e] != f[k][e]:
                                    a = False
                                    break
                        b = len(uqx) != 0
                        return a and b
                return False
        return False
        
if __name__ == "__main__":
    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    nmels = 128
    hop_length = 512

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    from tracking.logger import Logger
    from datasets.binjob import Binworker
    from multiprocessing.pool import ThreadPool
    
    worker = Binworker()
    logger = Logger()

    clip_dataset = CachedClippedDataset(
        worker=worker, 
        logger=logger,
        clip_nsamples = clip_length_samples,
        overlap_nsamples = clip_overlap_samples,
    )

    balancer = CachedDatasetBalancer(clip_dataset, logger=logger, worker=worker, verbose=True)
    limited_dataset = DatasetLimiter(clip_dataset, limit=42, balancer=balancer, randomize=False, balanced=True)

    tensordataset = TensorAudioDataset(
        dataset=limited_dataset,
        label_accessor=BinaryLabelAccessor(),
        feature_accessor=MelSpectrogramFeatureAccessor()
    )

    verifier = BinaryTensorDatasetVerifier(verbose=True)
    valid = verifier.verify(tensordataset)
    print(valid)

    # # Now without limiting
    # balanced = BalancedDatasetDecorator(clip_dataset, balancer=balancer)
    # tensordataset = TensorAudioDataset(
    #     balanced,
    #     label_accessor=BinaryLabelAccessor(),
    #     feature_accessor=MelSpectrogramFeatureAccessor()
    # )

    # valid, unique_features, label_stats = verifier.verify(tensordataset)
    # print(valid, len(unique_features), label_stats)

    