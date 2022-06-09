#!/usr/bin/env python3
import argparse
from gc import callbacks
import os
import sys
import pathlib
from typing import Optional, Iterable, Any, Dict, Tuple, Mapping

import git
import torch
import pytorch_lightning as pl
import torchmetrics

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from models.ResNet18.ResNet18 import ResNet18
from metrics import customwandbplots
from metrics.metriccollection import GliderMetrics

class ResNet18LightningWrapper(pl.LightningModule):
    def __init__(
        self, 
        learning_rate: float,
        weight_decay: float,
        betas: Iterable[float],
        *args: Any, 
        n_model_outputs: int = 2,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        class_names: Optional[Iterable[str]] = None,
        **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        self.activation = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss()
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.class_names = class_names
        
        self.model = ResNet18(n_outputs=n_model_outputs, output_activation=torch.nn.Sigmoid())

        self.val_metrics = GliderMetrics(num_classes=n_model_outputs, class_names=class_names)
        self.test_metrics = GliderMetrics(num_classes=n_model_outputs, class_names=class_names)
        # keep this separate for custom logging implementation for confusion matrix
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True, threshold=0.5) # Dont normalize here, send raw values to wandb and normalize in GUI
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True, threshold=0.5)

        self.save_hyperparameters()

    def log_metrics(self, step: str, metrics: Mapping[str, torch.Tensor], confusion: torch.Tensor) -> None:
        for metric, values in metrics.items():
            self.log(metric, values)
        
        # Compute and log confusion matrix
        biophonic_confusion = confusion[0]
        anthropogenic_confusion = confusion[1]
        
        self.logger.experiment.log({f"{step}_bio_confusion_matrix": customwandbplots.confusion_matrix(self.logger, biophonic_confusion, class_names=["not bio", "bio"], title=f"{step} confusion matrix (biophonic)")})
        self.logger.experiment.log({f"{step}_anth_confusion_matrix": customwandbplots.confusion_matrix(self.logger, anthropogenic_confusion, class_names=["not anth", "anth"], title=f"{step} confusion matrix (anthropogenic)")})

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Expect batch to have shape (batch_size, 1, n_mel_bands, n_time_frames)"""
        return self.model(X)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        X, Y = batch # [batch_size, 1, n_mels, n_time_frames], [batch_size, 2]
        Yhat = self.forward(X) # [batch_size, 2]
        loss = self.loss(Yhat, Y)
        self.log("train_loss", loss)
        return dict(loss=loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        X, Y = batch
        Yhat = self.forward(X)
        loss = self.loss(Yhat, Y)
        self.log("val_loss", loss)
        self.val_metrics.update(Yhat.float(), Y.int())
        self.val_confusion_matrix.update(Yhat.float(), Y.int())
        return dict(loss=loss)

    def validation_epoch_end(self, *args, **kwargs) -> None:
        metrics = self.val_metrics.compute(step="val")
        confusion = self.val_confusion_matrix.compute()
        
        self.log_metrics("val", metrics, confusion)
        
        self.val_metrics.reset()
        self.val_confusion_matrix.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        X, Y = batch
        Yhat = self.forward(X)
        loss = self.loss(Yhat, Y)
        self.log("test_loss", loss)

        self.test_metrics.update(Yhat.float(), Y.int())
        self.test_confusion_matrix.update(Yhat.float(), Y.int())
        return dict(loss=loss)

    def test_epoch_end(self, *args, **kwargs) -> None:
        metrics = self.test_metrics.compute("test")
        confusion = self.test_confusion_matrix.compute()
        self.log_metrics("test", metrics, confusion)
        self.test_metrics.reset()
        self.test_confusion_matrix.reset()

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=self.betas, weight_decay=self.weight_decay)
        return dict(optimizer=optimizer)