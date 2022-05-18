#!/usr/bin/env python3
import sys
import pathlib
from typing import Mapping, Iterable, Optional

import git
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from interfaces import ILoggerFactory, ITensorAudioDataset
from tracking.loggerfactory import LoggerFactory
from tracking.logger import BasicLogger

from datasets.initdata import create_tensorset

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ITensorAudioDataset, subset: Iterable[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.subset[index]]

class ClippedGliderDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        nfft: int,
        nmels: int,
        hop_length: int,
        clip_duration_seconds: float,
        clip_overlap_seconds: float,
        logger_factory: Optional[ILoggerFactory] = None,
        num_workers: Optional[int] = 0,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        
        self.batch_size = batch_size
        self.nfft = nfft
        self.nmels = nmels
        self.hop_length = hop_length
        self.clip_duration_seconds = clip_duration_seconds
        self.clip_overlap_seconds = clip_overlap_seconds
        self.logger_factory = logger_factory if logger_factory is not None else LoggerFactory(logger_type=BasicLogger)
        self.num_workers = num_workers

        tensorset, balancer = create_tensorset(
            logger_factory=self.logger_factory,
            nfft=self.nfft,
            nmels=self.nmels,
            hop_length=self.hop_length,
            clip_duration_seconds=self.clip_duration_seconds,
            clip_overlap_seconds=self.clip_overlap_seconds,
        )
        self.tensorset = tensorset
        self.balancer = balancer
        self._setup_done = False

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Given a dataset with examples that have any of the classes c1, c2, c3 or c4
        F.ex.: 
        ['c1', 'c1', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c2', 'c2', 'c2',
       'c2', 'c2', 'c2', 'c3', 'c3', 'c3', 'c3', 'c3', 'c3', 'c4', 'c4',
       'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4']
        
        With class distributions: c1: 15%, c2: 15%, c3: 10%, c4: 60%. c4 is heavily overrepresented compared to the other classes.
        We balance the dataset for training and validation, but keep the same original distributions during testing to avoid introducing bias into the evaluation
        
        Steps:
        1. Group examples by their class:
            c1 examples: ['c1', 'c1', 'c1', 'c1', 'c1'],
            c2 examples: ['c2', 'c2', 'c2', 'c2', 'c2', 'c2', 'c2', 'c2', 'c2'],
            c3 examples: ['c3', 'c3', 'c3', 'c3', 'c3', 'c3'],
            c4 examples: ['c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4'],
        2. Find the class with the least number of instances. 
            In this example c1 has the fewest examples out of all the classes, with 5 examples. 
            set min_size = 5
        3. We need to select a subset of the dataset such that all the classes are equally represented for training and validation, and to use the instaces not seen during training nor validation for testing. We can get a balanced subset by selecting an equal number of examples from each class group. However, this would result in there being no examples left in the least represented classes to use during testing. Therefore we select a fraction < 1.0 of min_size from each class group and reserve these instances for training and validation, the remaining instances are reserved for testing. 
            E.g.:
            train_val_reserve_fraction = 0.8
            n_instances_train_and_val = min_size * train_val_reserve_fraction = 4
            n_test_instances = len(dataset) - n_instances_train_and_val

            Resulting in the following subsets from each class group:
            c1 examples: ['c1', 'c1', 'c1', 'c1',]
            c2 examples: ['c2', 'c2', 'c2', 'c2',]
            c3 examples: ['c3', 'c3', 'c3', 'c3',]
            c4 examples: ['c4', 'c4', 'c4', 'c4',]

        4. We then split each class group of the balanced subset into individual 'train' and 'validation' subsets, aggregate the results from each subset.
            In this implementation 80% of the balanced subset is used for training, and the remaining 20% is used for validation.
            In this example, each subset class group has 4 instances, 80% of them are reserved for training, and the remaning 20% are reserved for validation. 
            4 * 0.8 = 3.2 ≈ 3
            4 * 0.2 = 0.8 ≈ 1
            
            The selection and aggregation would in this example be: 
            train c1 examples: ['c1', 'c1', 'c1']
            train c2 examples: ['c2', 'c2', 'c2']
            train c3 examples: ['c3', 'c3', 'c3']
            train c4 examples: ['c4', 'c4', 'c4']
            final train subset: ['c1', 'c1', 'c2', 'c2', 'c3', 'c3', 'c4', 'c4']

            val c1 examples: ['c1']
            val c2 examples: ['c2']
            val c3 examples: ['c3']
            val c4 examples: ['c4']
            final validation subset: ['c1', 'c2', 'c3', 'c4']
        5. All of the remaining instances from all class groups are aggregated and used for testing. 
            In this example the remaining instances are:
            test c1 examples: ['c1']
            test c2 examples: ['c2', 'c2', 'c2', 'c2', 'c2']
            test c3 examples: ['c3', 'c3']
            test c4 examples: ['c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4']
            
            final testing subset: 
            ['c1', 'c2', 'c2', 'c2', 'c2', 'c2', 'c3', 'c3', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4', 'c4']
        Done! We now have balanced train and validtion subsets of the original dataset and a testing subset where the label distributions are unchanged.
        """
        
        distributions = self.balancer.label_distributions()
        for key in distributions.keys():
            distributions[key] = np.array(distributions[key])

        min_size = np.min([len(class_indeces) for class_indeces in distributions.values()], axis=0)
        self.balanced_indeces = np.concatenate([np.random.choice(indeces, size=min_size, replace=False) for indeces in distributions.values()])
        self.remaining_indeces = np.concatenate([indeces[np.where(~np.isin(indeces, self.balanced_indeces))[0]] for indeces in distributions.values()])
        
        train_val_percentage = 0.8
        n_for_training = int(len(self.balanced_indeces) * train_val_percentage)

        # Indeces used for training and validation
        self.train_and_val_part = np.random.choice(self.balanced_indeces, n_for_training, replace=False)
        self.train_indeces, self.val_indeces = train_test_split(self.train_and_val_part, test_size=0.2)

        # Indeces for testing
        test_part = self.balanced_indeces[np.where(~np.isin(self.balanced_indeces, self.train_and_val_part))[0]] # The indeces from "balanced" that was not used for the train nor val sets
        self.test_indeces = np.concatenate([test_part, self.remaining_indeces]) # This way label distribution is maintained for testset
        # Train-, val- and testsets as subset datasets
        self.train = SubsetDataset(dataset=self.tensorset, subset=self.train_indeces) # These are balanced
        self.val = SubsetDataset(dataset=self.tensorset, subset=self.val_indeces) # These are balanced
        self.test = SubsetDataset(dataset=self.tensorset, subset=self.test_indeces) # These are unbalanced
        dataloader_sizes = [len(self.train), len(self.val), len(self.test)]
        class_group_sizes = [len(indeces) for indeces in distributions.values()]

        assert np.sum(dataloader_sizes) == np.sum(class_group_sizes)
        assert np.sum(dataloader_sizes) == len(self.tensorset)

        self._setup_done = True

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def loggables(self) -> Mapping[str, any]:
        if not self._setup_done:
            self.setup()

        distributions = self.balancer.label_distributions()
        n_per_label = {label: len(indeces) for label, indeces in distributions.items()}

        to_log = {
            "loader_sizes": {
                "train_loader_size": len(self.train),
                "val_loader_size": len(self.val),
                "test_loader_size": len(self.test)
            },
            "label_distributions": n_per_label,
            "tensorset_size": len(self.tensorset)
        }
        return to_log

    def get_tensor_audio_dataset(self):
        return self.tensorset


if __name__ == "__main__":
    dataset = ClippedGliderDataModule(
        batch_size=8,
        nfft=1024,
        nmels=128,
        hop_length=512,
        clip_duration_seconds=10.0,
        clip_overlap_seconds=4.0
    )
    print(dataset)
    dataset.setup()
    from rich import print
    print(dataset.loggables())