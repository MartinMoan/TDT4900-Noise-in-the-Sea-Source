#!/usr/bin/env python3
import argparse
from contextlib import ExitStack
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
dataset_dir = config.HOME_PROJECT_DIR.joinpath("datasets")
if not dataset_dir.exists():
    dataset_dir.mkdir(parents=False, exist_ok=False)
import torchvision

class FashionModel(pl.LightningModule):
    def __init__(self, n_inputs, n_outputs) -> None:
        super().__init__()
        self.l = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_inputs, n_outputs),
            torch.nn.Softmax(dim=0)
        )
        self.lossfunc = torch.nn.NLLLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=n_outputs)

    def forward(self, X):
        return self.l(X)

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self.lossfunc(Yhat, Y)
        self.accuracy(Yhat, Y)
        self.log("train_accuracy", self.accuracy, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def test_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self.lossfunc(Yhat, Y)
        self.accuracy(Yhat, Y)
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

class FashionDataset(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        root = dataset_dir.joinpath("torchvision")
        if not root.exists():
            root.mkdir(parents=False, exist_ok=False)
        
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=root, 
            download=True,
            train=True,
            target_transform=FashionDataset.to_label_tensor,
            transform=torchvision.transforms.ToTensor(),
        )

        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=root, 
            download=True,
            train=False, 
            target_transform=FashionDataset.to_label_tensor,
            transform=torchvision.transforms.ToTensor(),
        )
        self.batch_size = batch_size

    def to_label_tensor(Y):
        return torch.tensor(Y).type(torch.LongTensor)
    
    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())
        print(dl, dl.dataset)
        return dl
    
    def test_dataloader(self):
        dl = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())
        print(dl, dl.dataset)
        return dl

def main(hyperparams):
    batch_size = hyperparams.batch_size
    dataset = FashionDataset(batch_size=batch_size)
    
    image_shape = (28, 28)
    in_features = np.prod(image_shape)
    out_features = 10 # class index/label
    model = FashionModel(n_inputs=in_features, n_outputs=out_features)

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
        logger=logger,
        max_epochs=10
        # auto_scale_batch_size=True # Not supported for DDP per. vXXX: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#batch-size-finder
    )
    
    logger.watch(model)

    # trainer.tune(model, datamodule=dataset)
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)

def init():
    parser = argparse.ArgumentParser()
    # Data params
    parser.add_argument("-batch_size", type=int, required=True)
    # Tracking params
    parser.add_argument("-tracking_name", type=str, required=True)
    parser.add_argument("-tracking_note", type=str, required=False)
    parser.add_argument("-tracking_tags", type=str, nargs="+", required=False)
    # Other params
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("-num_gpus", type=int, required=True)
    parser.add_argument("-num_nodes", type=int, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    from pprint import pprint
    pprint(os.environ)
    hyperparams = init()
    main(hyperparams)
'''
Run with:

python experiment.py -batch_size 16 -tracking_name "Experiment" -tracking_note "Testing FashionMNIST model to ensure distributed training on SLURM cluster works." --verbose -num_gpus 8 -num_nodes 4
'''
