#!/usr/bin/env python3
import sys
import pathlib
from typing import Iterable, NewType
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
        logger_factory: ILoggerFactory,
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
        model_size: str=ModelSize.base384,
        verbose=True) -> None:

        super().__init__()
        self._ast = ASTWrapper(
            logger_factory=logger_factory,
            activation_func=torch.nn.Sigmoid(),
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
        self._printlogger = logger_factory.create_logger()
        self._batch_size = batch_size
        self._accuracy = torchmetrics.Accuracy(num_classes=2)
        self._aucroc = torchmetrics.AUROC(num_classes=2)
        self._precision = torchmetrics.Precision(num_classes=2)
        self._recall = torchmetrics.Recall(num_classes=2)
        self._average_precision = torchmetrics.AveragePrecision(num_classes=2)
        self._f1 = torchmetrics.F1Score(num_classes=2)
        self._confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True, threshold=0.5) # Dont normalize here, send raw values to wandb and normalize in GUI
        self._verbose = verbose
        self.save_hyperparameters()

    def update_metrics(self, stepname, Yhat, Y):
        self._accuracy(Yhat.float(), Y.int())
        self._aucroc.update(Yhat.float(), Y.int())
        self._precision.update(Yhat.float(), Y.int())
        self._recall.update(Yhat.float(), Y.int())
        self._average_precision.update(Yhat.float(), Y.int())
        self._f1.update(Yhat.float(), Y.int())
        
        self.log(f"{stepname}_accuracy", self._accuracy, on_step=False, on_epoch=True)
        self.log(f"{stepname}_aucroc", self._aucroc, on_step=False, on_epoch=True)
        self.log(f"{stepname}_precision", self._precision, on_step=False, on_epoch=True)
        self.log(f"{stepname}_recall", self._recall, on_step=False, on_epoch=True)
        self.log(f"{stepname}_average_precision", self._average_precision, on_step=False, on_epoch=True)
        self.log(f"{stepname}_f1", self._f1, on_step=False, on_epoch=True)

        self._confusion_matrix.update(Yhat.float(), Y.int())

    def log_confusion_matrix(self, stepname):
        confusion = self._confusion_matrix.compute() # has shape (2, 2, 2)
        biophonic_confusion = confusion[0]
        anthropogenic_confusion = confusion[1]
        
        self.logger.experiment.log({f"{stepname}_bio_confusion_matrix": customwandbplots.confusion_matrix(biophonic_confusion, class_names=["not bio", "bio"], title=f"{stepname} confusion matrix (biophonic)")})
        self.logger.experiment.log({f"{stepname}_anth_confusion_matrix": customwandbplots.confusion_matrix(anthropogenic_confusion, class_names=["not anth", "anth"], title=f"{stepname} confusion matrix (anthropogenic)")})

    def forward(self, X):
        """Expect batch to have shape (batch_size, 1, n_mel_bands, n_time_frames)"""
        # AST.py expects input to have shape (batch_size, n_time_fames, n_mel_bans), swap third and fourth axis of X and squeeze second axis
        return self._ast(X)

    def training_step(self, batch, batch_idx):
        X, Y = batch # [batch_size, 1, n_mels, n_time_frames], [batch_size, 2]
        Yhat = self.forward(X) # [batch_size, 2]
        loss = self._lossfunc(Yhat, Y)
        self.log("train_loss", loss)
        return dict(loss=loss) # these are sent as input to training_epoch_end    

    def test_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self._lossfunc(Yhat, Y)
        self.update_metrics("test", Yhat, Y)
        return dict(loss=loss)

    def test_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix("test")

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self._lossfunc(Yhat, Y)
        self.update_metrics("val", Yhat, Y)
        return dict(loss=loss)
    
    def validation_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix("val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self._learning_rate, betas=self._betas, weight_decay=self._weight_decay)
        return dict(optimizer=optimizer)