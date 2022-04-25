#!/usr/bin/env python3
import multiprocessing
import sys
import pathlib
from typing import Iterable, Mapping
from datetime import datetime, timedelta

import git
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ITrainer, ITensorAudioDataset, ILoggerFactory, IOptimizerProvider
from tools.typechecking import verify

class Trainer(ITrainer):
    # @verify
    def __init__(
        self, 
        logger_factory: ILoggerFactory, 
        optimizer_provider: IOptimizerProvider,
        batch_size: int,
        epochs: int,
        lossfunction: torch.nn.modules.loss._Loss,
        num_workers: int = multiprocessing.cpu_count(),
        device: str = None
        ) -> None:
        super().__init__()

        self.logger = logger_factory.create_logger()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.optimizer_provider = optimizer_provider
        self.lossfunction = lossfunction
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self, 
        model: torch.nn.Module, 
        dataset_indeces: Iterable[int], 
        dataset: ITensorAudioDataset) -> torch.nn.Module:

        sampler = SubsetRandomSampler(dataset_indeces)

        trainset = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            sampler=sampler, 
            num_workers=self.num_workers
        )

        optimizer = self.optimizer_provider.provide(model)

        # if from_checkpoint is not None:
        #     if type(from_checkpoint) == bool and from_checkpoint:
        #         model, optimizer, _ = load(model, optimizer, locals())
        #     elif type(from_checkpoint) == str or type(from_checkpoint) == pathlib.PosixPath:
        #         model, optimizer, _ = load(model, optimizer, locals(), checkpoint_dir=from_checkpoint)

        last_print = None
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch, (X, Y) in enumerate(trainset):
                X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
                X, Y = X.to(self.device), Y.to(self.device)
                Yhat = model(X).type(torch.FloatTensor)
                
                optimizer.zero_grad()

                loss = self.lossfunction(Yhat, Y)
                
                loss.backward()
                optimizer.step()

                current_loss = loss.item()

                if np.isnan(current_loss):
                    raise Exception("Loss is nan...")
                
                epoch_loss += current_loss
                if last_print is None or datetime.now() - last_print >= timedelta(seconds=config.PRINT_INTERVAL_SECONDS):
                    self.logger.log(f"Training epoch {epoch + 1} / {self.epochs} batch {batch + 1} / {len(trainset)} loss {current_loss:.5f}")
                    last_print = datetime.now()

            average_epoch_loss = epoch_loss / len(trainset)
            # checkpoint(_started_at, checkpoint_td, model, optimizer, locals())
            # saver.save(model, mode="training", avg_epoch_loss=average_epoch_loss)
        self.logger.log("Train iterations complete!")
        return model

    @property
    def properties(self) -> Mapping[str, any]:
        out = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "optimizer_provider": self.optimizer_provider.properties,
            "lossfunction": self.lossfunction,
            "device": self.device,
        }
        return out