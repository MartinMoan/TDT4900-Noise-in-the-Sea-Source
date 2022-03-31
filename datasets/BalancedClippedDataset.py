#!/usr/bin/env python3
import warnings
import abc
from typing import Iterable, Mapping, Tuple
import sys
from matplotlib.pyplot import cla
import sklearn
import torch
from ICustomDataset import ICustomDataset
from glider.audiodata import LabeledAudioData
from glider.clipping import ClippedDataset
from ITensorAudioDataset import ITensorAudioDataset, ILabelAccessor, IFeatureAccessor, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
import multiprocessing
from multiprocessing import Pool
import math
import numpy as np
from rich import print

class IBalancedDataset(ITensorAudioDataset, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train() -> None:
        """Set the dataset to training mode: meaning that the dataset will be balanced"""
    
    @abc.abstractmethod
    def eval() -> None:
        """Set the dataset to eval mode, the dataset ouput will not be balanced."""


class BalancedDataset(IBalancedDataset):
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
    def __init__(self, dataset: ICustomDataset, label_accessor: ILabelAccessor, feature_accessor: IFeatureAccessor) -> None:
        super().__init__()
        self._dataset = dataset
        
        self._split_labels: Mapping[str, Iterable[int]] = self._split_by_labels(dataset)

        self._both_labels_indeces: Iterable[int] = self._split_labels["both"]
        self._neither_labels_indeces: Iterable[int] = self._split_labels["neither"]
        self._anthropogenic_indeces: Iterable[int] = self._split_labels["anthropogenic"]
        self._biophonic_indeces: Iterable[int] = self._split_labels["biophonic"]

        average_num_examples_per_label: float = np.mean([len(self._both_labels_indeces), len(self._anthropogenic_indeces), len(self._biophonic_indeces)])
        _num_unlabeled_to_use = min(len(self._neither_labels_indeces), int(average_num_examples_per_label))

        
        self._unlabeled_indeces_to_use = np.random.choice(self._neither_labels_indeces, size=_num_unlabeled_to_use, replace=False)
        self._labeled_indeces = np.concatenate([self._both_labels_indeces, self._anthropogenic_indeces, self._biophonic_indeces], axis=0)
        self._indeces_to_use = np.concatenate((self._labeled_indeces, self._unlabeled_indeces_to_use), axis=0)

        self._label_accessor = label_accessor
        self._feature_accessor = feature_accessor

    def _split_by_labels_poolfunc(self, dataset: ICustomDataset, start: int, end: int) -> Mapping[str, Iterable[int]]:
        proc = multiprocessing.current_process()

        both = []
        anthropogenic = []
        biophonic = []
        neither = []
        
        for i in range(start, min(end, len(dataset))):
            part = math.ceil((end - start) * 0.025)
            if (i - start) % part == 0:
                percentage = ((i - start) / (end - start)) * 100
                print(f"Balancing: {proc.name} {percentage:.2f}%")
                
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

    def __len__(self) -> int:
        return len(self._indeces_to_use)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Returns a tuple (index: int, X: torch.Tensor, Y: torch.Tensor)"""
        audio_data = self._dataset[index]
        features = torch.nan_to_num(self._feature_accessor(audio_data), nan=0, posinf=10, neginf=-10)
        labels = self._label_accessor(audio_data, features)
        return index, features, labels

    def eval() -> None:
        pass

    def train() -> None:
        pass
    # def classes(self) -> Mapping[str, int]:
    #     return self._dataset.classes()

    def example_shape(self) -> tuple[int, ...]:
        if len(self) > 0:
            audio_data = self._dataset[0]
            features = self._feature_accessor(audio_data)
            return features.shape
        else:
            warnings.warn("The TensorAudioDataset has length 0, this will likely cause unexpected results")
            return None

    def label_shape(self) -> tuple[int, ...]:
        if len(self) > 0:
            audio_data = self._dataset[0]
            features = self._feature_accessor(audio_data)
            labels = self._label_accessor(audio_data, features)
            return labels.shape
        else:
            warnings.warn("The TensorAudioDataset has length 0, this will likely cause unexpected results")
            return None

def print_label_distribution_stats(dataset: BalancedDataset):
    print(f"Num both antrhopogenic and biogenic: {len(balanced_dataset._both_labels_indeces)}")
    print(f"Num just anthropogenic: {len(balanced_dataset._anthropogenic_indeces)}")
    print(f"Num just biogenic: {len(balanced_dataset._biophonic_indeces)}")
    print(f"Num unlabeled: {(len(balanced_dataset._neither_labels_indeces))}")

import sklearn
import sklearn.model_selection
class CustomKFolder(sklearn.model_selection.KFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, balanced_dataset: IBalancedDataset):
        super().sp
        return super().split(X, y, groups)

        

if __name__ == "__main__":
    # clip_duration_seconds = 10.0
    # clip_overlap_seconds = 2.0
    # dataset = ClippedDataset(clip_duration_seconds=clip_duration_seconds, clip_overlap_seconds=clip_overlap_seconds)
    # balanced_dataset = BalancedDataset(
    #     dataset, 
    #     feature_accessor=MelSpectrogramFeatureAccessor(), 
    #     label_accessor=BinaryLabelAccessor()
    # )

    from sklearn.model_selection import KFold
    folds = KFold(n_splits=5, shuffle=True)
    balanced_dataset = [chr(i) for i in range(100, 200)]
    print(folds)
    for fold, (training_indeces, test_indeces) in enumerate(folds.split(balanced_dataset)):
        
    exit()

    

    print_label_distribution_stats(balanced_dataset)