#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
from turtle import forward
from typing import Mapping, Iterable, Optional
import multiprocessing

import git
import torch
import torch.utils.data
import torchmetrics
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from interfaces import ILoggerFactory, ITensorAudioDataset, IDatasetBalancer
from tracking.logger import BasicLogger
from tracking.loggerfactory import LoggerFactory

from models.AST.ASTWrapper import ASTWrapper

from datasets.tensordataset import TensorAudioDataset

from experiments.AST_pretrained.initdata import create_tensorset

from metrics import customwandbplots
from tracking.datasettracker import track_dataset

class AstLightningWrapper(pl.LightningModule):
    """
    AST (Audio Spectrogram Transformer) pretraining wrapper. Enables custom activation
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
        fstride=10, 
        tstride=10, 
        input_fdim=128, 
        input_tdim=1024, 
        imagenet_pretrain=True, 
        audioset_pretrain=False, 
        model_size='base384',
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
    def __init__(self, tensorset: TensorAudioDataset, balancer: IDatasetBalancer, batch_size: int, train_limit: int = None, val_limit: int = None, test_limit: int = None) -> None:
        super().__init__()
        self.tensorset = tensorset
        self.balancer = balancer
        self.batch_size = batch_size
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.val_limit = val_limit

    def setup(self, stage: Optional[str] = None) -> None:
        self.balancer.shuffle()

        self.eval_only_indeces = self.balancer.eval_only_indeces()
        self.train_indeces = self.balancer.train_indeces()

        distributions = self.balancer.label_distributions()
        n_per_label = {label: len(indeces) for label, indeces in distributions.items()}


        train_val_percentage = 0.8
        test_percentage = 1 - train_val_percentage
        
        n_for_training = int(len(self.train_indeces) * train_val_percentage)
        n_from_eval_only = int(len(self.eval_only_indeces) * test_percentage)

        # Indeces used for training and validation
        self.train_and_val_part = self.train_indeces[:n_for_training]
        self.train_indeces, self.val_indeces = train_test_split(self.train_and_val_part, test_size=0.2)

        # Indeces for testing
        test_part = self.train_indeces[n_for_training:] # These are balanced
        unbalanced_parts = self.eval_only_indeces[:n_from_eval_only] # These are unbalanced
        self.test_indeces = np.concatenate([test_part, unbalanced_parts]) # This way label distribution is maintained for testset

        # Train-, val- and testsets as subset datasets
        self.train = SubsetDataset(dataset=self.tensorset, subset=self.train_indeces, limit=self.train_limit)
        self.val = SubsetDataset(dataset=self.tensorset, subset=self.val_indeces, limit=self.val_limit)
        self.test = SubsetDataset(dataset=self.tensorset, subset=self.test_indeces, limit=self.test_limit)

        to_log = {
            "loader_sizes": {
                "train_loader_size": len(self.train),
                "val_loader_size": len(self.val),
                "test_loader_size": len(self.test)
            },
            "label_distributions": n_per_label,
            "tensorset_size": len(self.tensorset)
        }
        wandb.config.update(to_log)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

def main(hyperparams):
    sr = 128000
    fdim = hyperparams.nmels
    tdim = int((hyperparams.clip_duration_seconds * sr / hyperparams.hop_length) + 1)

    logger_factory = LoggerFactory(logger_type=BasicLogger)
    mylogger = logger_factory.create_logger()
    mylogger.log("Received hyperparams:", vars(hyperparams))

    logger = WandbLogger(
        # name=hyperparams.tracking_name,
        save_dir=str(config.HOME_PROJECT_DIR.absolute()),
        offline=False,
        project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
        entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
    )

    model = AstLightningWrapper(
        logger_factory=logger_factory,
        learning_rate=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
        betas=hyperparams.betas,
        batch_size=hyperparams.batch_size, # Only required for auto_scaling of batch_size
        activation_func=None,
        n_model_outputs=2,
        fstride=hyperparams.fstride,
        tstride=hyperparams.tstride,
        input_fdim=fdim,
        input_tdim=tdim,
        imagenet_pretrain=hyperparams.imagenet_pretrain,
        audioset_pretrain=hyperparams.audioset_pretrain,
        model_size=hyperparams.model_size,
        verbose=hyperparams.verbose,
    )

    tensorset, balancer = create_tensorset(
        logger_factory=logger_factory,
        nfft=hyperparams.nfft,
        nmels=hyperparams.nmels,
        hop_length=hyperparams.hop_length,
        clip_duration_seconds=hyperparams.clip_duration_seconds,
        clip_overlap_seconds=hyperparams.clip_overlap_seconds,
    )

    dataset = ClippedGliderDataModule(
        tensorset=tensorset, 
        balancer=balancer,
        batch_size=hyperparams.batch_size,
        train_limit=None,
        test_limit=None,
        val_limit=None,
    )

    trainer = pl.Trainer(
        # accelerator="gpu", 
        # devices=hyperparams.num_gpus, 
        # num_nodes=hyperparams.num_nodes,
        # strategy="ddp",
        max_epochs=hyperparams.epochs,
        logger=logger,
        # auto_scale_batch_size=True # Not supported for DDP per. vXXX: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#batch-size-finder
    )
    
    logger.watch(model)
    wandb.config.update(vars(hyperparams))
    track_dataset(tensorset, n_examples=50)

    # trainer.tune(model, datamodule=dataset)
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)

def init():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument("-weight_decay", type=float, required=True)
    parser.add_argument("-learning_rate", type=float, required=True)
    parser.add_argument("-betas", type=float, nargs="+", required=True)
    parser.add_argument("-fstride", type=int, default=10)
    parser.add_argument("-tstride", type=int, default=10)
    parser.add_argument("--imagenet_pretrain", action=argparse.BooleanOptionalAction)
    parser.add_argument("--audioset_pretrain", action=argparse.BooleanOptionalAction)
    parser.add_argument("-model_size", type=str, choices=["tiny224", "small224", "base224", "base384"])

    # Data params
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-nmels", type=int, required=True)
    parser.add_argument("-nfft", type=int, required=True)
    parser.add_argument("-hop_length", type=int, required=True)
    parser.add_argument("-clip_duration_seconds", type=float, required=True)
    parser.add_argument("-clip_overlap_seconds", type=float, required=True)

    # Training params
    parser.add_argument("-epochs", type=int, required=True)
    parser.add_argument("-kfolds", type=int, required=True)
    
    # Tracking params
    parser.add_argument("-tracking_name", type=str, required=True)
    parser.add_argument("-tracking_note", type=str, required=True)
    parser.add_argument("-tracking_tags", type=str, nargs="+", required=True)
    parser.add_argument("-track_n_examples", type=int, default=50)
    # Other params
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("-num_gpus", type=int, required=True)
    parser.add_argument("-num_nodes", type=int, required=True)

    # parser.add_argument("-verification_dataset_limit", type=int, default=42)
    # parser.add_argument("-proper_dataset_limit", default=0.7)
    args = parser.parse_args()
    args.betas = tuple(args.betas)
    return args

if __name__ == "__main__":
    hyperparams = init()
    main(hyperparams)
'''
Run with:

python ast_lightning.py -batch_size 16 -epochs 3 -learning_rate 0.0001 -weight_decay 5e-7 -betas 0.95 0.999 -kfolds 5 -nmels 128 -hop_length 512 -nfft 2046 -fstride 10 -tstride 10 -model_size base384 -clip_duration_seconds 10.0 -clip_overlap_seconds 4.0 -tracking_name "AST ImageNet Pretrained" -tracking_note "AST pretrained on ImageNet (but not AudioSet, to enable differing input shapes)" -tracking_tags "AST" "ImageNet" "No-AudioSet" --imagenet_pretrain --no-audioset_pretrain --verbose
'''
