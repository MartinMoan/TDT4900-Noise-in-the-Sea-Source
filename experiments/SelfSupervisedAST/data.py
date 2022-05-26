#!/usr/bin/env python3
from copy import copy
import pathlib
import sys
from typing import Mapping, Optional, Tuple, Iterable, Any

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
from datasets.tensordataset import TensorAudioDataset

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
        train_part: Optional[float] = 0.8,
        val_part: Optional[float] = 0.2,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):

        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        assert train_part <= 1.0 and train_part > 0.0
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
        self.train_part = train_part
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

    def setstaging(self, stage: str):
        assert stage in [TrainingStage.finetune.value, TrainingStage.pretrain.value]
        self.stage = stage
    
    def pretrain(self) -> None:
        self.stage = TrainingStage.pretrain.value
    
    def finetune(self) -> None:
        self.stage = TrainingStage.finetune.value

    def subset_distributions(self, distributions: Mapping[str, np.ndarray]) -> Mapping[str, SubsetDataset]:
        min_size = np.min([len(class_indeces) for class_indeces in distributions.values()], axis=0)
        balanced_indeces = np.concatenate([np.random.choice(indeces, size=min_size, replace=False) for indeces in distributions.values()])
        np.random.shuffle(balanced_indeces)
        remaining_indeces = np.concatenate([indeces[np.where(~np.isin(indeces, balanced_indeces))[0]] for indeces in distributions.values()])
        np.random.shuffle(remaining_indeces)
        n_for_training = int(len(balanced_indeces) * self.train_part)

        # Indeces used for training and validation
        train_and_val_part = np.random.choice(balanced_indeces, n_for_training, replace=False)
        train_indeces, val_indeces = train_test_split(train_and_val_part, test_size=self.val_part)
        
        np.random.shuffle(train_indeces)
        np.random.shuffle(val_indeces)

        # Indeces for testing
        test_part = balanced_indeces[np.where(~np.isin(balanced_indeces, train_and_val_part))[0]] # The indeces from "balanced" that was not used for the train nor val sets
        test_indeces = np.concatenate([test_part, remaining_indeces]) # This way label distribution is maintained for testset
        np.random.shuffle(test_indeces)
        # Train-, val- and testsets as subset datasets
        train = SubsetDataset(dataset=self.tensorset, subset=train_indeces) # These are balanced
        val = SubsetDataset(dataset=self.tensorset, subset=val_indeces) # These are balanced
        test = SubsetDataset(dataset=self.tensorset, subset=test_indeces) # These are unbalanced
        return dict(train=train, test=test, val=val)

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
        
        expected_total_samples = np.sum([len(indeces) for indeces in raw_distributions.values()])
        dists = {key: len(indeces) for key, indeces in raw_distributions.items()}
        assert expected_total_samples == len(self.tensorset), f"The total number of samples after balancing does not equal the number of classes in tensorset, expected total {expected_total_samples} balanced samples (raw class distributions: {dists}) but tensorset has length {len(self.tensorset)}"

        self.pretraining_subsets = {}
        self.finetuning_subsets = {}

        for key, indeces in raw_distributions.items():
            size = round(len(indeces) * self.pretext_part)
            for_pretraining = np.random.choice(indeces, size=size, replace=False)
            for_finetuning = indeces[np.where(~np.isin(indeces, for_pretraining))]

            self.pretraining_subsets[key] = for_pretraining
            self.finetuning_subsets[key] = for_finetuning

        all_indeces = np.concatenate([indeces for indeces in self.pretraining_subsets.values()] + [indeces for indeces in self.finetuning_subsets.values()])
        unique, counts = np.unique(all_indeces, return_counts=True)
        dup = unique[counts > 1]
        assert len(dup) == 0, f"There are samples that are duplicated across the pretrain/finetune train, val, test subsets: {dup}"

        self.pretraining_dataloaders = self.subset_distributions(self.pretraining_subsets)
        self.finetuning_dataloaders = self.subset_distributions(self.finetuning_subsets)

        n_in_ft_ttv = [len(subset) for subset in self.finetuning_dataloaders.values()]
        n_in_pt_ttf = [len(subset) for subset in self.pretraining_dataloaders.values()]

        total_for_ft = [len(indeces) for indeces in self.finetuning_subsets.values()]
        total_for_pt = [len(indeces) for indeces in self.pretraining_subsets.values()]
        exp_ft_part_of_tensorset = round(len(self.tensorset) * (1 - self.pretext_part))
        exp_pt_part_of_tensorset = round(len(self.tensorset) * self.pretext_part)
        
        # pytorch lightning will suplicate some samples to ensure dataloader have a whole number of batches to provide. 
        # The loader will duplicate R number of samples equal to: len(loader) % batch_size
        # With P=3 loaders for pretraining, and F=3 loaders for fine-tuning, the maximum number of duplicated samples are:
        # (self.batch_size - 1) * (F + P) = T
        # So we should not require that the total number of samples in all the dataloaders equal the length of the original dataset.
        # But that the total number of samples in all dataloaders be within a tolerance of T samples from the number of samples in the original dataset
        TOLERANCE = (self.batch_size - 1) * (len(self.pretraining_dataloaders.values()) + len(self.finetuning_dataloaders.values()))

        assert np.abs(np.sum(n_in_ft_ttv) - np.sum(total_for_ft)) <= TOLERANCE
        assert np.abs(np.sum(n_in_pt_ttf) - np.sum(total_for_pt)) <= TOLERANCE
        assert np.abs(np.sum(total_for_ft) - exp_ft_part_of_tensorset) <= TOLERANCE
        assert np.abs(np.sum(total_for_pt) == exp_pt_part_of_tensorset) <= TOLERANCE

        self._setup_done = True

    def get_subset(self, stage: str) -> None:
        if self.is_pretrain_stage:
            return self.pretraining_dataloaders[stage]
        else:
            return self.finetuning_dataloaders[stage]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        subset = self.get_subset("train")
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        subset = self.get_subset("val")
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        subset = self.get_subset("test")
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size, num_workers=self.num_workers)

    def loggables(self) -> Mapping[str, Any]:
        if not self._setup_done:
            self.setup()

        distributions = self.balancer.label_distributions()
        n_per_label = {label: len(indeces) for label, indeces in distributions.items()}

        original_staging = copy(self.stage)

        self.pretrain()
        ptrain, pval, ptest = self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
        self.finetune()
        ftrain, fval, ftest = self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

        to_log = {
            "loader_sizes": {
                "pretrain": {
                    "train_loader_size": len(ptrain),
                    "val_loader_size": len(pval),
                    "test_loader_size": len(ptest)
                },
                "finetune": {
                    "train_loader_size": len(ftrain),
                    "val_loader_size": len(fval),
                    "test_loader_size": len(ftest)
                },
            },
            "pretrain_distributions": {key: len(indeces) for key, indeces in self.pretraining_subsets.items()},
            "finetune_distributions": {key: len(indeces) for key, indeces in self.finetuning_subsets.items()},
            "label_distributions": n_per_label,
            "tensorset_size": len(self.tensorset)
        }
        self.setstaging(original_staging)
        return to_log

    def get_tensor_audio_dataset(self) -> TensorAudioDataset:
        return self.tensorset

    def class_names(self) -> Iterable[str]:
        classes = self.tensorset.classes()
        output = np.array(list(classes.keys()))
        order = list(classes.values())
        return output[order]

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

    print(len(dataset.train_dataloader()))

    dataset.finetune()
    
    print(len(dataset.train_dataloader()))

    dataset.pretrain()
    print(len(dataset.train_dataloader()))
