#!/usr/bin/env python3
import sys
import pathlib
import multiprocessing
from datetime import datetime, timedelta
from typing import Iterable, Tuple, Mapping

import git
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import IEvaluator, ITensorAudioDataset, ILoggerFactory

class Evaluator(IEvaluator):
    def __init__(self, logger_factory: ILoggerFactory, batch_size: int, num_workers: int = multiprocessing.cpu_count(), device: str = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger_factory.create_logger()

    def evaluate(
        self, 
        model: torch.nn.Module, 
        dataset_indeces: Iterable[int], 
        dataset: ITensorAudioDataset) -> Tuple[torch.Tensor, torch.Tensor]:

        testset = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            sampler=SubsetRandomSampler(dataset_indeces), 
            num_workers=self.num_workers)

        def cat(t1, t2):
            if t1 is None:
                return t2
            return torch.cat((t1, t2))

        last_print = None
        model.eval()
        with torch.no_grad():
            truth = None
            predictions = None
            for index, (X, Y) in enumerate(testset):
                X, Y = X.to(self.device), Y.to(self.device)
                Yhat = model(X)
                
                Yhat = Yhat.cpu()
                X, Y = X.cpu(), Y.cpu()

                truth = cat(truth, Y)
                predictions = cat(predictions, Yhat)
                
                if last_print is None or (datetime.now() - last_print) >= timedelta(seconds=config.LOG_INTERVAL_SECONDS):
                    self.logger.log(f"Eval index {index} / {len(testset)}")
                    last_print = datetime.now()
            self.logger.log("Eval iterations complete!")
            return truth, predictions

    @property
    def properties(self) -> Mapping[str, any]:
        props = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
        }
        return props
