#!/usr/bin/env python3
from cProfile import label
import multiprocessing
from multiprocessing.sharedctypes import Value
import pathlib
import sys
import abc
from typing import Tuple, Mapping, Iterable, Union
from multiprocessing import Pool
import math
from unittest import result

import git
import numpy as np
from rich import print
REPO_DIR=pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)
sys.path.insert(0, str(REPO_DIR))

import config
from ITensorAudioDataset import ITensorAudioDataset
from clipping import ClippedDataset, ClippingCacheDecorator
from ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
from ITensorAudioDataset import FileLengthTensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from limiting import DatasetLimiter
from IDatasetBalancer import DatasetBalancer, BalancerCacheDecorator, BalancedDatasetDecorator
from logger import Logger, ILogger

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

class IDatasetVerifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def verify(self, dataset: ITensorAudioDataset) -> Tuple[bool, Mapping[str, any]]:
        raise NotImplementedError

class BinaryTensorDatasetVerifier(IDatasetVerifier):
    def __init__(self, verbose: bool = True, logger: ILogger = Logger()) -> None:
        self._verbose = verbose
        self.logger = logger

    def _verify(self, dataset: ITensorAudioDataset, start: int, end: int) -> Tuple[set, Iterable[ValueCount]]:
        proc = multiprocessing.current_process()
        unique_label_values = []
        unique_feature_values = set([])
        for i in range(start, end):
            index, X, Y = dataset[i]
            feature_values = set(np.unique(X.numpy()))
            unique_feature_values = unique_feature_values.union(feature_values)

            values = list(Y.numpy())

            valuecount = ValueCount(values, count=1)
            unique_label_values = self._flatten([valuecount], output=unique_label_values)
            if self._verbose:
                percentage = ((i - start) / (end - start)) * 100
                part = math.ceil((end - start) * 0.05)
                if (i - start) % part == 0:
                    self.logger.log(f"VerificationWorker PID {proc.pid} - {percentage:.2f}%")
        return unique_feature_values, unique_label_values

    def _flatten(self, existing: Iterable[ValueCount], output: Iterable[ValueCount] = []) -> Iterable[ValueCount]:
        for valuecount in existing:
            if valuecount in output:
                index = output.index(valuecount)
                element = output[index]
                element.count += valuecount.count
            else:
                output.append(valuecount)
        return output

    def binjob_async(iterable: Iterable, function: callable, function_args: Tuple[any] = (), function_kwargs: Mapping[str, any] = {}) -> Iterable[any]:
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            tasks = []
            binsize = math.ceil(len(iterable) / multiprocessing.cpu_count())
            for start in range(0, len(iterable), binsize):
                end = start + binsize
                task = pool.apply_async(function, args=(iterable, start, end, *function_args), kwds=function_kwargs)
                tasks.append(task)
            
            results = [task.get() for task in tasks]
            return results

    def _valid_label_stats(self, label_value_counts: Iterable[ValueCount]) -> Tuple[bool, str]:
        # total = np.sum([valuecount.count for valuecount in label_value_counts])
        message = "Label values are valid!"
        valid = True
        if len(label_value_counts) != 4:
            valid = False
            message = f"\n\tThere are not 4 unique types of label/class pairs ([0, 0], [0, 1], [1, 0] and [1, 1]). Found {len(label_value_counts)}"
        
        count = None
        for valuecount in label_value_counts:
            if valuecount.count == 0:
                valid = False
                message += f"\n\tThere are no instances of the label/class pair {valuecount.values} - {valuecount}"
            if count is None:
                count = valuecount.count
            elif valuecount.count != count:
                valid = False
                message += f"\n\tThe number of label/class pairs is not equal for all label/class pairs. Expected {count} instances for all label/class pairs but found {valuecount.count} instances of {valuecount.values}"
        if not valid:
            message = "The label values are invalid!" + message
        return valid, message

    def _valid_feature_values(self, unique_features: set) -> Tuple[bool, str]:
        valid = (len(unique_features) != 0)
        message = "Features are valid: The number of unique feature values is not 0"
        if valid:
            message = f"Features are invalid! : The number of unique feature values is 0 ({len(unique_features)})"
        return valid, message

    def _getstats(self, dataset: ITensorAudioDataset) -> Tuple[set, Iterable[ValueCount]]:
        results = BinaryTensorDatasetVerifier.binjob_async(dataset, self._verify)
        label_value_counts = []
        unique_feature_values = set([])
        for index, (feature_values, label_values) in enumerate(results):
            if self._verbose:
                self.logger.log(f"Verification aggregation - {((index + 1) / len(results) * 100.0):.1f}%")
            label_value_counts = self._flatten(label_values, label_value_counts)
            unique_feature_values = unique_feature_values.union(feature_values)
        
        return unique_feature_values, label_value_counts
            
    def verify(self, dataset: ITensorAudioDataset) -> Tuple[bool, set, Iterable[ValueCount]]:
        unique_features, label_value_counts = self._getstats(dataset)
        valid_labels, labels_status_message = self._valid_label_stats(label_value_counts=label_value_counts)
        valid_features, features_status_message = self._valid_feature_values(unique_features)
        valid = (valid_labels and valid_features)
        if not valid: 
            error_msg = f"The dataset verifier {self.__class__.__name__} could not verify the dataset {dataset.__class__.__name__}."
            error_msg += f"Features values status:\n{features_status_message}"
            error_msg += f"Label values status:\n{labels_status_message}"
            raise Exception(error_msg)
        return valid, unique_features, label_value_counts
        
if __name__ == "__main__":
    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    nmels = 128
    hop_length = 512

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    clip_dataset = ClippingCacheDecorator(
        clip_nsamples = clip_length_samples, 
        overlap_nsamples = clip_overlap_samples
    )

    limited_dataset = DatasetLimiter(clip_dataset, limit=42, randomize=False, balanced=True)

    tensordataset = FileLengthTensorAudioDataset(
        dataset=limited_dataset,
        label_accessor=BinaryLabelAccessor(),
        feature_accessor=MelSpectrogramFeatureAccessor()
    )

    verifier = BinaryTensorDatasetVerifier(verbose=True)
    valid, unique_features, label_stats = verifier.verify(tensordataset)
    print(valid)