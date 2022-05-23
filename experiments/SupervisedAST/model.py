#!/usr/bin/env python3
import sys
import pathlib
from typing import Iterable, Optional
from enum import Enum

import git
import torch
import torch.utils.data
import torchmetrics
import pytorch_lightning as pl

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from interfaces import ILoggerFactory
from models.AST.ASTWrapper import ASTWrapper
from metrics import customwandbplots
from metrics.metriccollection import GliderMetrics

class ModelSize(Enum):
    tiny224 = "tiny224"
    small224 = "small224"
    base224 = "base224"
    base384 = "base384"

class AstLightningWrapper(pl.LightningModule):
    """
    AST (Audio Spectrogram Transformer) Pytorch Lightning module. Enables PytorchLightning to manage the model training, evaluation and testing.
    In this way, the model train, eval and test steps and the model itself can be "hardware agnostic". Enabling us to run it on as few or as many GPUs and nodes we might need. 
    """
    def __init__(
        self, 
        learning_rate: float,
        weight_decay: float,
        betas: Iterable[float],
        batch_size: int,
        activation_func: torch.nn.Module = None, 
        n_model_outputs=2, 
        fstride: int=10, 
        tstride: int=10, 
        input_fdim: int=128, 
        input_tdim: int=1024, 
        imagenet_pretrain: bool=True, 
        audioset_pretrain: bool=False, 
        model_size: str = ModelSize.base384.value,
        class_names: Optional[Iterable[str]] = None,
        verbose=True) -> None:

        super().__init__()
        self._ast = ASTWrapper(
            activation_func=None, # Because we use BCEWithLogitsLoss
            label_dim=n_model_outputs, 
            fstride=fstride, 
            tstride=tstride, 
            input_fdim=input_fdim, 
            input_tdim=input_tdim, 
            imagenet_pretrain=imagenet_pretrain, 
            audioset_pretrain=audioset_pretrain, 
            model_size=model_size, 
            verbose=verbose
        )
        self._activation = activation_func
        self._lossfunc = torch.nn.BCEWithLogitsLoss()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._betas = betas
        self._batch_size = batch_size

        self.metrics = GliderMetrics(num_classes=n_model_outputs, class_names=class_names)
        # keep this separate for custom logging implementation for confusion matrix
        self._confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True, threshold=0.5) # Dont normalize here, send raw values to wandb and normalize in GUI
        self._verbose = verbose
        self.save_hyperparameters()

    def update_metrics(self, stepname, Yhat, Y):
        self.metrics.update(Yhat.float(), Y.int())
        self._confusion_matrix.update(Yhat.float(), Y.int())

    def log_metrics(self, step):
        # Log general metrics
        metrics = self.metrics.compute(step)
        for metric, values in metrics.items():
            self.log(metric, values)
        
        # Compute and log confusion matrix
        confusion = self._confusion_matrix.compute() # has shape (2, 2, 2)
        biophonic_confusion = confusion[0]
        anthropogenic_confusion = confusion[1]
        
        self.logger.experiment.log({f"{step}_bio_confusion_matrix": customwandbplots.confusion_matrix(self.logger, biophonic_confusion, class_names=["not bio", "bio"], title=f"{step} confusion matrix (biophonic)")})
        self.logger.experiment.log({f"{step}_anth_confusion_matrix": customwandbplots.confusion_matrix(self.logger, anthropogenic_confusion, class_names=["not anth", "anth"], title=f"{step} confusion matrix (anthropogenic)")})        

    def forward(self, X):
        """Expect batch to have shape (batch_size, 1, n_mel_bands, n_time_frames)"""
        # AST.py expects input to have shape (batch_size, n_time_fames, n_mel_bans), swap third and fourth axis of X and squeeze second axis
        return self._ast(X)

    def forward_batch(self, step, batch, batch_idx):
        X, Y = batch # [batch_size, 1, n_mels, n_time_frames], [batch_size, 2]
        Yhat = self.forward(X) # [batch_size, 2]
        loss = self._lossfunc(Yhat, Y)
        self.log("loss", loss)
        if step != "train":
            self.update_metrics(step, Yhat, Y)
        return dict(loss=loss)

    def training_step(self, batch, batch_idx):
        return self.forward_batch("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.forward_batch("val", batch, batch_idx)

    def validation_epoch_end(self, outputs) -> None:
        self.log_metrics("val")

    def test_step(self, batch, batch_idx):
        return self.forward_batch("test", batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        self.log_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self._ast.parameters(), lr=self._learning_rate, betas=self._betas, weight_decay=self._weight_decay)
        return dict(optimizer=optimizer)

if __name__ == "__main__":
    m = AstLightningWrapper(
        learning_rate=0.1,
        weight_decay=0.5,
        betas=[0.1, 0.2],
        batch_size=8,
    )
    m.configure_optimizers()