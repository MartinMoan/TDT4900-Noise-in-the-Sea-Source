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
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from interfaces import ILoggerFactory, IModelProvider, IAsyncWorker, ICustomDataset, ITensorAudioDataset
from tracking.logger import Logger, LogFormatter
from tracking.loggerfactory import LoggerFactory

from models.AST.ASTWrapper import ASTWrapper
from models.AST.AST import ASTModel

from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import DatasetBalancer, CachedDatasetBalancer
from datasets.binjob import Binworker
from datasets.tensordataset import BinaryLabelAccessor, MelSpectrogramFeatureAccessor, TensorAudioDataset

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
        self._ast = ASTModel(
            logger_factory=logger_factory,
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
        self.lossfunc = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.printlogger = logger_factory.create_logger()
        self.batch_size = batch_size
        self.accuracy = torchmetrics.Accuracy(num_classes=2)
        self.save_hyperparameters()

    def forward(self, X):
        X = X.permute(0, 1, 3, 2)
        X = X.squeeze(dim=1)
        return self._ast(X)

    def training_step(self, batch, batch_idx):
        """Expect batch to have shape (batch_size, 1, n_mel_bands, n_time_frames)"""
        # AST.py expects input to have shape (batch_size, n_time_fames, n_mel_bans), swap third and fourth axis of X and squeeze second axis
        X, Y = batch # [batch_size, 1, n_mels, n_time_frames], [batch_size, 2]
        Yhat = self.forward(X) # [batch_size, 2]
        loss = self.lossfunc(Yhat, Y)
        self.accuracy(Yhat, Y.int())
        self.log("train_accuracy", self.accuracy, on_step=False, on_epoch=True)
        return dict(loss=loss) # these are sent as input to training_epoch_end    

    def test_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self.lossfunc(Yhat, Y)
        self.accuracy(Yhat, Y.int())
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self.lossfunc(Yhat, Y)
        self.accuracy(Yhat, Y.int())
        self.log("val_accuracy", self.accuracy, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, betas=self.betas, weight_decay=self.weight_decay)
        return dict(optimizer=optimizer)

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ITensorAudioDataset, subset: Iterable[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.subset[index]]

def create_tensorset(logger_factory, nfft, nmels, hop_length, clip_duration_seconds, clip_overlap_seconds):
    clips = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=Binworker(),
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds
    )

    label_accessor = BinaryLabelAccessor()
    feature_accessor = MelSpectrogramFeatureAccessor(
        logger_factory=logger_factory,
        n_mels=nmels,
        n_fft=nfft,
        hop_length=hop_length,
        scale_melbands=False,
        verbose=True
    )

    tensorset = TensorAudioDataset(
        dataset=clips,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor,
        logger_factory=logger_factory
    )

    balancer = CachedDatasetBalancer(
        dataset=clips,
        logger_factory=logger_factory,
        worker=Binworker(),
        verbose=True
    )

    eval_only_indeces = balancer.eval_only_indeces()
    train_indeces = balancer.train_indeces()

    return tensorset, eval_only_indeces, train_indeces

class ClippedGliderDataModule(pl.LightningDataModule):
    def __init__(self, tensorset: TensorAudioDataset, eval_only_indeces: Iterable[int], train_indeces: Iterable[int], batch_size: int) -> None:
        super().__init__()
        self.tensorset = tensorset
        self.eval_only_indeces = eval_only_indeces[:10] # TODO: Remove limit
        self.train_indeces = train_indeces[:10] # TODO: Remove limit
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_and_val_part = np.random.choice(self.train_indeces, size=int(len(self.train_indeces * 0.8))) 
        test_part = self.train_indeces[~np.isin(self.train_indeces, self.train_and_val_part)]
        self.test_indeces = np.concatenate([test_part, self.eval_only_indeces]) # this part for testing only, not seen anytime during training nor validation
        self.train_indeces, self.val_indeces = train_test_split(self.train_and_val_part, test_size=0.2)
        
        self.train = SubsetDataset(dataset=self.tensorset, subset=self.train_indeces)
        self.val = SubsetDataset(dataset=self.tensorset, subset=self.val_indeces)
        self.test = SubsetDataset(dataset=self.tensorset, subset=self.test_indeces)

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

    logger_factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(),
        logger_kwargs=dict(logformatter=LogFormatter())
    )
    logger = logger_factory.create_logger()
    logger.log("Received hyperparams:", vars(hyperparams))

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

    tensorset, eval_only_indeces, train_indeces = create_tensorset(
        logger_factory=logger_factory,
        nfft=hyperparams.nfft,
        nmels=hyperparams.nmels,
        hop_length=hyperparams.hop_length,
        clip_duration_seconds=hyperparams.clip_duration_seconds,
        clip_overlap_seconds=hyperparams.clip_overlap_seconds,
    )

    dataset = ClippedGliderDataModule(
        tensorset=tensorset, 
        eval_only_indeces=eval_only_indeces, 
        train_indeces=train_indeces,
        batch_size=hyperparams.batch_size
    )

    logger = WandbLogger(
        name=hyperparams.tracking_name,
        save_dir=str(config.HOME_PROJECT_DIR.absolute()),
        offline=False,
        project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
        entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
    )

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=hyperparams.num_gpus, 
        num_nodes=hyperparams.num_nodes,
        strategy="ddp",
        max_epochs=hyperparams.epochs,
        logger=logger,
        # auto_scale_batch_size=True # Not supported for DDP per. vXXX: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#batch-size-finder
    )
    
    logger.watch(model)

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
