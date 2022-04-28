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
from interfaces import ITrainer, ITensorAudioDataset, ILoggerFactory, IOptimizerProvider, ITracker, IMetricComputer
from tools.typechecking import verify

class Trainer(ITrainer):
    # @verify
    def __init__(
        self, 
        logger_factory: ILoggerFactory, 
        optimizer_provider: IOptimizerProvider,
        tracker: ITracker,
        metric_computer: IMetricComputer,
        batch_size: int,
        epochs: int,
        lossfunction: torch.nn.modules.loss._Loss,
        num_workers: int = multiprocessing.cpu_count(),
        device: str = None) -> None:
        super().__init__()
        
        self.tracker = tracker
        self.logger = logger_factory.create_logger()
        self.metric_computer = metric_computer
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

        percentage = (len(dataset_indeces) / len(dataset)) * 100.0
        self.logger.log(f"Beginning training iterations using {len(dataset_indeces)} / {len(dataset)} dataset instances ({percentage:.2f}%)")
        
        sampler = SubsetRandomSampler(dataset_indeces)

        trainset = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            sampler=sampler, 
            num_workers=self.num_workers
        )

        optimizer = self.optimizer_provider.provide(model)

        last_print = None
        model.train()
        for epoch in range(self.epochs):
            cumulative_epoch_loss = 0
            # self.logger.log(f"Beginning epoch {epoch + 1} / {self.epochs}...")
            for batch, (X, Y) in enumerate(trainset):
                # self.logger.log(f"Start batch {batch + 1} / {len(trainset)} in epoch {epoch + 1} / {self.epochs}...")
                X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
                X, Y = X.to(self.device), Y.to(self.device)
                Yhat = model(X).type(torch.FloatTensor).to(self.device)
                
                optimizer.zero_grad()

                # metrics = self.metric_computer(Y, Yhat)

                loss = self.lossfunction(Yhat, Y)
                
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()

                self.tracker.track(
                    {
                        "epoch": epoch, 
                        "batch_loss": batch_loss
                    }
                )

                if np.isnan(batch_loss):
                    raise Exception("Loss is nan...")
                
                cumulative_epoch_loss += batch_loss
                
                if last_print is None or datetime.now() - last_print >= timedelta(seconds=config.PRINT_INTERVAL_SECONDS):
                    self.logger.log(f"Training epoch {epoch + 1} / {self.epochs} batch {batch + 1} / {len(trainset)} loss {batch_loss:.5f}")
                    last_print = datetime.now()
            
            avg_epoch_loss = cumulative_epoch_loss / len(trainset)
            self.tracker.track(
                {
                    "loss": avg_epoch_loss, 
                    "epoch": epoch
                }
            )
            # self.logger.log("Epoch complete!")
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