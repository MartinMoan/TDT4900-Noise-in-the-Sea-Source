#!/usr/bin/env python3
import pathlib
import sys
from typing import Mapping, Optional, Tuple

import git
import pytorch_lightning as pl
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from experiments.SelfSupervisedAST.model import TrainingStage
from datasets.datamodule import SubsetDataset
from datasets.initdata import create_tensorset

class SelfSupervisedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        nfft: int,
        nmels: int,
        hop_length: int,
        clip_duration_seconds: float,
        clip_overlap_seconds: float,
        pretext_part: float = 0.8,
        stage: Optional[str] = TrainingStage.pretrain.value, 
        num_workers: Optional[int] = 0,
        train_test_part: Tuple[float, float] = (0.8, 0.2),
        val_part: float = 0.2,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):

        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        assert len(train_test_part) == 2
        assert np.sum(train_test_part) <= 1.0
        assert val_part <= 1.0 and val_part > 0.0

        self.stage = stage
        self.batch_size = batch_size
        self.nfft = nfft
        self.nmels = nmels
        self.hop_length = hop_length
        self.clip_duration_seconds = clip_duration_seconds
        self.clip_overlap_seconds = clip_overlap_seconds
        self.num_workers = num_workers
        self.pretext_part = pretext_part
        self.train_test_part = train_test_part
        self.val_part = val_part

        tensorset, balancer = create_tensorset(
            nfft=self.nfft,
            nmels=self.nmels,
            hop_length=self.hop_length,
            clip_duration_seconds=self.clip_duration_seconds,
            clip_overlap_seconds=self.clip_overlap_seconds,
        )
        self.tensorset = tensorset
        self.balancer = balancer
        self._setup_done = False
        
    @property
    def is_pretrain_stage(self) -> bool:
        return (self.stage == TrainingStage.pretrain.value)
    
    def pretrain(self) -> None:
        self.stage = TrainingStage.pretrain.value
    
    def finetune(self) -> None:
        self.stage = TrainingStage.finetune.value

    def subset_distributions(self, distributions: Mapping[str, np.ndarray]) -> Tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
        min_size = np.min([len(class_indeces) for class_indeces in distributions.values()], axis=0)
        balanced_indeces = np.concatenate([np.random.choice(indeces, size=min_size, replace=False) for indeces in distributions.values()])
        remaining_indeces = np.concatenate([indeces[np.where(~np.isin(indeces, balanced_indeces))[0]] for indeces in distributions.values()])
        
        train_val_percentage, test_percentage = self.train_test_part
        n_for_training = int(len(balanced_indeces) * train_val_percentage)

        # Indeces used for training and validation
        train_and_val_part = np.random.choice(balanced_indeces, n_for_training, replace=False)
        train_indeces, val_indeces = train_test_split(train_and_val_part, test_size=self.val_part)

        # Indeces for testing
        test_part = balanced_indeces[np.where(~np.isin(balanced_indeces, train_and_val_part))[0]] # The indeces from "balanced" that was not used for the train nor val sets
        test_indeces = np.concatenate([test_part, remaining_indeces]) # This way label distribution is maintained for testset
        # Train-, val- and testsets as subset datasets
        train = SubsetDataset(dataset=self.tensorset, subset=train_indeces) # These are balanced
        val = SubsetDataset(dataset=self.tensorset, subset=val_indeces) # These are balanced
        test = SubsetDataset(dataset=self.tensorset, subset=test_indeces) # These are unbalanced
        return train, test, val

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train, val and test subsets for both pretraining and finetuning
        We assume that pytorch_lightning.seed_everything() has been called, such that random operations are relicated equally in all processes/ranks.

        In pretraining, we do not care about the labels themselves, but we use them to ensure that we don't introduce bias into the known-class distrubutions during pretraining
        E.g. we take an equal percentage of the clips from each class/label group. We call this data the 'pretraining subset'

        During pretraining we also have train, val and test loops as usual, all of which require different subsets of the data.
        So we split the pretraining subset into train, val and test subsets, following the same distrubutions as during fine-tuning. 

        The remaining data is used during fine-tuning, we call this subset the 'finetuning subset'.

        For the finetuning subset, we ensure that the class/label group distributions are balanced during training and evaluation, but unaffected during testing.

        Args:
            stage (Optional[str], optional): A stage string provided by PytorchLightning, is not used. Defaults to None.
        """
        raw_distributions = self.balancer.label_distributions()
        for key in raw_distributions.keys():
            raw_distributions[key] = np.array(raw_distributions[key])

        self.pretraining_subsets = {}
        self.finetuning_subsets = {}
        import math
        for key, indeces in raw_distributions.items():
            print(indeces)
            for_pretraining = np.random.choice(indeces, size=math.floor(len(indeces) * self.pretext_part), replace=False)
            for_finetuning = [indeces[np.where(~np.isin(indeces, for_pretraining))]]

            self.pretraining_subsets[key] = for_pretraining
            self.finetuning_subsets[key] = for_finetuning

        self.fsubsets = self.subset_distributions(self.finetuning_subsets)
        assert np.sum([len(dataset) for dataset in self.fsubsets]) == np.sum([len(indeces) for indeces in self.finetuning_subsets.values()])
        assert np.sum([len(dataset) for dataset in self.fsubsets]) == np.floor(len(self.tensorset) * (1 - self.pretext_part))
        
        self.psubsets = self.subset_distributions(self.pretraining_subsets)
        assert np.sum([len(dataset) for dataset in self.psubsets]) == np.sum([len(indeces) for indeces in self.pretraining_subsets.values()])
        assert np.sum([len(dataset) for dataset in self.psubsets]) == np.floor(len(self.tensorset) * self.pretext_part)

        self._setup_done = True

    def get_subset(self, stage: str) -> None:
        keys = {"train": 0, "val": 1, "test": 2}
        if self.is_pretrain_stage:
            return self.psubsets[keys[stage]]
        else:
            return self.fsubsets[keys[stage]]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        subset = self.get_subset("train")
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        subset = self.get_subset("val")
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        subset = self.get_subset("test")
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":
    dataset = SelfSupervisedDataModule(
        batch_size=8, 
        nfft=2048, 
        nmels=128, 
        hop_length=512, 
        clip_duration_seconds=10.0, 
        clip_overlap_seconds=4.0
    )
    dataset.setup()
    print(dataset)

    print(dataset.stage)
    print(dataset.train_dataloader())