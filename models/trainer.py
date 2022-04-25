#!/usr/bin/env python3
import sys
import pathlib
from typing import Iterable, Mapping
from inspect import stack

import git
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ITrainer, ITensorAudioDataset, ILoggerFactory

class Trainer(ITrainer):
    def __init__(self, logger_factory: ILoggerFactory, batch_size: int, num_workers: int, optimizer) -> None:
        super().__init__()
        self.logger = logger_factory.create_logger()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optimizer = optimizer

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

        optimizer = optimizer_ref(model.parameters(), lr=lr, weight_decay=weight_decay)
        lossfunction = loss_ref()

        if from_checkpoint is not None:
            if type(from_checkpoint) == bool and from_checkpoint:
                model, optimizer, _ = load(model, optimizer, locals())
            elif type(from_checkpoint) == str or type(from_checkpoint) == pathlib.PosixPath:
                model, optimizer, _ = load(model, optimizer, locals(), checkpoint_dir=from_checkpoint)

        last_print = None
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch, (index, X, Y) in enumerate(trainset):
                X, Y = X.to(device), Y.to(device)
                Yhat = model(X)
                optimizer.zero_grad()

                loss = lossfunction(Yhat, Y)
                
                loss.backward()
                optimizer.step()

                current_loss = loss.item()

                if np.isnan(current_loss):
                    raise Exception("Loss is nan...")
                
                epoch_loss += current_loss
                if last_print is None or datetime.now() - last_print >= timedelta(seconds=config.PRINT_INTERVAL_SECONDS):
                    logger.log(f"\t\ttraining epoch {epoch} batch {batch} / {len(trainset)} loss {current_loss}")
                    last_print = datetime.now()

            average_epoch_loss = epoch_loss / len(trainset)
            checkpoint(_started_at, checkpoint_td, model, optimizer, locals())
            # saver.save(model, mode="training", avg_epoch_loss=average_epoch_loss)
        logger.log("Train iterations complete!")
        return model, optimizer
        return model

    @property
    def properties(self) -> Mapping[str, any]:
        out = {}
        return out