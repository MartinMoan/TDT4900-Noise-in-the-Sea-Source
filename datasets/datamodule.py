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
    def __init__(self, dataset: ITensorAudioDataset, subset: Iterable[int], limit: int = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.limit = limit if limit is not None else len(subset)
        self.subset = subset[:self.limit]

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
        logger_factory: ILoggerFactory=None,
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

    def setup(self, stage: Optional[str] = None) -> None:
        self.eval_only_indeces = self.balancer.eval_only_indeces()
        self.train_indeces = self.balancer.train_indeces()

        train_val_percentage = 0.8
        test_percentage = 1 - train_val_percentage
        
        n_for_training = int(len(self.train_indeces) * train_val_percentage)
        n_from_eval_only = int(len(self.eval_only_indeces) * test_percentage)

        # Indeces used for training and validation
        self.train_and_val_part = np.random.choice(self.train_indeces, n_for_training)
        self.train_indeces, self.val_indeces = train_test_split(self.train_and_val_part, test_size=0.2)

        # Indeces for testing
        test_part = self.train_indeces[n_for_training:] # These are balanced
        unbalanced_parts = self.eval_only_indeces[:n_from_eval_only] # These are unbalanced
        self.test_indeces = np.concatenate([test_part, unbalanced_parts]) # This way label distribution is maintained for testset

        # Train-, val- and testsets as subset datasets
        self.train = SubsetDataset(dataset=self.tensorset, subset=self.train_indeces, limit=self.train_limit)
        self.val = SubsetDataset(dataset=self.tensorset, subset=self.val_indeces, limit=self.val_limit)
        self.test = SubsetDataset(dataset=self.tensorset, subset=self.test_indeces, limit=self.test_limit)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size) 

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size)

    def loggables(self) -> Mapping[str, any]:
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