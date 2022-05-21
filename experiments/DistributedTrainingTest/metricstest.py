#!/usr/bin/env python3
import os
import enum
import pathlib
import sys
from turtle import forward
from typing import Any, Optional, Tuple, Union, Collection, Sequence, Callable, Mapping

import git
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.utils.data
import torchmetrics
from torchmetrics import MetricCollection, Metric
import logging
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from metrics import multilabelmap

class RandomDummyModel(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.layer = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        # if self.layer is None:
        #     self.n_inputs = np.prod(X.shape[1:])
        #     self.layer = torch.nn.Linear(self.n_inputs, self.n_outputs)
        flattened = X.flatten(start_dim=1)
        assert flattened.shape == (B, self.n_inputs), f"input has incorrect shape, expected product of individual samples in the batch to have {self.n_inputs} number of elements, but received input with shape {X.shape} assuming batch size {B} yielding {flattened.shape[1]} number of elements in batch"
        return self.layer(flattened)

class RandomDummyModule(pl.LightningModule):
    def __init__(
        self, 
        input_shape: Tuple[int, ...], 
        output_shape: Tuple[int, ...],
        loss: Union[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        metrics: Union[torchmetrics.MetricCollection, torchmetrics.Metric]) -> None:
        
        super().__init__()
        self.model = RandomDummyModel(np.prod(input_shape), np.prod(output_shape))
        self.loss = loss
        self.metrics = metrics
        self.input_shape = input_shape
        self.output_shape = output_shape

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(params=self.model.parameters(), lr=0.0005)
        return dict(optimizer=optimizer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def forward_batch(self, batch: torch.Tensor, bathc_idx: int) -> Mapping[str, Any]:
        X, Y = batch
        Yhat = self.forward(X)
        self.metrics.update(Yhat, Y)
        loss = self.loss(Yhat, Y.float())
        self.log("loss", loss)
        return dict(loss=loss)

    def log_metrics(self, step: str):
        self.metrics.prefix = step + "_"
        metrics = self.metrics.compute()
        for name, values in metrics.items():
            if isinstance(values, torch.Tensor):
                metrics[name] = np.array(values.detach().cpu().squeeze().numpy())
            elif type(values) == list:
                metrics[name] = np.array(values)
        
        C = self.output_shape[0]
        for name, values in metrics.items():
            print(name, values)
            if values.shape == self.output_shape:
                # metric is computed per class
                for c in range(C):
                    c_metric = torch.Tensor([values[c]])
                    key = f"{name}_c{c}"
                    self.log(key, c_metric)
            elif values.shape == ():
                # Single metric for all classes)
                self.log(name, torch.Tensor(values))

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, Any]:
        return self.forward_batch(batch, batch_idx)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, Any]:
        return self.forward_batch(batch, batch_idx)

    def validation_epoch_end(self, outputs: Mapping[str, Any]) -> None:
        self.log_metrics("val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, Any]:
        return self.forward_batch(batch, batch_idx)
    
    def test_epoch_end(self, outputs: Mapping[str, Any]) -> None:
        self.log_metrics("test")

class TaskType(enum.Enum):
    MULTI_LABEL = 1
    MULTI_CLASS = 2
    BINARY = 3
    REGRESSION = 4
    OTHER = -1

class RandomDummyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_shape: Tuple[int, ...], 
        num_classes: int = None, 
        task: TaskType = TaskType.MULTI_LABEL,
        output_shape: Tuple[int, ...] = None,
        min_output_value: Union[float, int] = 0.0,
        max_output_value: Union[float, int] = 1.0,
        size: int = 42) -> None:

        super().__init__()
        assert isinstance(task, TaskType), f"task argument has incorrect type, expected {type(TaskType)} but received object with type {type(task)}"

        if task in [TaskType.REGRESSION, TaskType.OTHER] and (output_shape is None or output_shape == ()):
            raise ValueError(f"Input task {task} requires the output shape to be known, but no valid output_shape was provided.")
        
        if task in [TaskType.MULTI_LABEL, TaskType.MULTI_CLASS, TaskType.BINARY] and num_classes is None:
            raise ValueError(f"Input task {task} requires the number of classes to be known but num_values argument was not provided")

        self.task = task
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.min = min_output_value
        self.max = max_output_value
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.rand(self.input_shape)
        if self.task == TaskType.MULTI_LABEL:
            # Output random integer tensor with shape (num_classes,) with binary integer values (0 or 1) for all elements
            # In this case multiple (or no) elements in the output can be 1
            Y = torch.randint(0, self.num_classes, self.output_shape).int()
        elif self.task == TaskType.MULTI_CLASS:
            # Output one-hot encoded binary integer tensor with shape (num_classes,)
            # In this case only a single (randomly selected) element of the output can be 1, all other elements will be 0
            cls = np.random.randint(0, self.num_classes)
            values = np.zeros((self.num_classes,))
            values[cls] = 1
            Y = torch.Tensor(values).int()
        elif self.taks == TaskType.BINARY:
            # Return random binary integer tensor with a single value being either 0 or 1.
            # E.g. output shape will be (1)
            Y = torch.randint(0, 2, size=1)
        elif self.task in [TaskType.REGRESSION, TaskType.OTHER]:
            # Output random float tensor with output_shape and values in range [min_value, max_value)
            Y = (self.min - self.max) * torch.rand(self.output_shape) + self.max
        return X, Y
        
    def __len__(self) -> int:
        return self.size

class RandomDummyDatamodule(pl.LightningDataModule):
    def __init__(
        self, 
        input_shape: Tuple[int, ...], 
        num_classes: int,
        batch_size: int,
        output_shape: Tuple[int, ...] = None,
        task: TaskType = TaskType.MULTI_LABEL,
        train_size: int = 4200,
        val_size: int = 420,
        test_size: int = 42, 
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_classes = num_classes
        self.task = task
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size

    def setup(self, *args, **kwargs) -> None:
        self.train = RandomDummyDataset(self.input_shape, num_classes=self.num_classes, task=self.task, output_shape=self.output_shape, size=self.train_size)
        self.val = RandomDummyDataset(self.input_shape, num_classes=self.num_classes, task=self.task, output_shape=self.output_shape, size=self.val_size)
        self.test = RandomDummyDataset(self.input_shape, num_classes=self.num_classes, task=self.task, output_shape=self.output_shape, size=self.test_size)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.val, batch_size=self.batch_size)
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size)

def main():
    C = 2
    average = "weighted" # 'macro', 'micro', 'weighted', None
    input_shape = (1, 128, 1024)
    output_shape = (C,)
    batch_size = 8
    
    logger = WandbLogger(
        save_dir=str(config.HOME_PROJECT_DIR.absolute()),
        offline=False,
        project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
        entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
        tags=["Dev", "DummyModel"],
        notes="Testing MetricCollection with custom MeanAveragePrecision implementation",
    )

    collection = MetricCollection(
        torchmetrics.Accuracy(num_classes=C, average=average),
        torchmetrics.AUROC(num_classes=C, average=average),
        torchmetrics.Precision(num_classes=C, average=average),
        torchmetrics.Recall(num_classes=C, average=average),
        torchmetrics.AveragePrecision(num_classes=C, average=average),
        torchmetrics.F1Score(num_classes=C, average=average),
        multilabelmap.MeanAveragePrecision(num_classes=C)
    )

    datamodule = RandomDummyDatamodule(
        input_shape=input_shape, 
        num_classes=C,
        output_shape=output_shape,
        batch_size=batch_size
    )
    
    trainer = pl.Trainer(accelerator="cpu", logger=logger, max_epochs=3)
    model = RandomDummyModule(
        input_shape=input_shape, 
        output_shape=output_shape,
        metrics=collection,
        loss=torch.nn.BCEWithLogitsLoss()
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
