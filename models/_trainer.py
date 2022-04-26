#!/usr/bin/env python3
import sys
import pathlib
from datetime import datetime, timedelta
import traceback
import warnings
import json
from typing import Iterable, Mapping, Type, Union

import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from rich import print
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score
import git
import numpy as np

import saver

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from checkpointer import checkpoint, load, clear_checkpoints
import tracker
from ITensorAudioDataset import ITensorAudioDataset
from IMetricComputer import IMetricComputer
from verifier import IDatasetVerifier
from logger import ILogger, Logger
from tracker import ITracker, Tracker
from modelprovider import IModelProvider

_started_at = datetime.now()

def verify_arguments(
    model_provider: IModelProvider,
    dataset: ITensorAudioDataset, 
    dataset_verifier: IDatasetVerifier,
    device: str, 
    kfolds: int = 5, 
    batch_size: int = 8, 
    num_workers: int = 1):
    
    model = model_provider.instantiate()
    
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"The return type of the model_provider.instantiate() has invalid type, must be torch.nn.Module instance, but received {type(model)}")
    
    if not isinstance(dataset, ITensorAudioDataset):
        raise Exception("Argument dataset does not point to a BasicDataset object")

    if kfolds < 0:
        raise Exception("Argument kfolds has negative value")
    if kfolds > 10:
        warnings.warn(f"Argument kfolds has excessively large value ({kfolds}), this will probably make training time very long.")
    
    if batch_size <= 0:
        raise Exception(f"Argument batch_size has invalid value ({batch_size}), must be positive integer greater than 0")
    
    if num_workers <= 0:
        raise Exception(f"Argument num_workers has invalid value ({num_workers}), must be positive integer greater than 0")

    dataset_verifier.verify(dataset)

def eval(
    model: torch.nn.Module, 
    dataset: ITensorAudioDataset, 
    test_samples, 
    batch_size: int, 
    num_workers: int, 
    device: str = None, 
    logger: ILogger = Logger()):
    
    testset = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_samples), num_workers=num_workers)

    if config.ENV == "dev":
        label_shape = dataset.label_shape()
        output_shape = (len(testset), *label_shape)
        zeros = np.zeros(output_shape)
        return zeros, zeros
    
    def cat(t1, t2):
        if t1 is None:
            return t2
        return torch.cat((t1, t2))

    last_print = None
    model.eval()
    with torch.no_grad():
        truth = None
        predictions = None
        for i, (index, X, Y) in enumerate(testset):
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            Yhat = Yhat.cpu()
            X, Y = X.cpu(), Y.cpu()

            truth = cat(truth, Y)
            predictions = cat(predictions, Yhat)
            if last_print is None or datetime.now() - last_print >= timedelta(seconds=config.PRINT_INTERVAL_SECONDS):
                logger.log(f"Eval index {i} / {len(testset)} - testset index {index}")
                last_print = datetime.now()
        logger.log("Eval iterations complete!")
        return truth, predictions

def train(
    model: torch.nn.Module, 
    dataset: ITensorAudioDataset,
    training_samples: Iterable[int],
    batch_size: int = 8,
    num_workers: int = 16,
    epochs: int = 8, 
    lr: float = 1e-3, 
    weight_decay: float = 1e-5, 
    from_checkpoint: Union[bool, pathlib.PosixPath] = False,
    loss_ref: Type[torch.nn.Module] = torch.nn.BCELoss, 
    optimizer_ref: Type[torch.nn.Module] = torch.optim.Adamax, 
    device: str = None,
    checkpoint_td: timedelta = timedelta(seconds=2),
    logger: ILogger = Logger()):
    
    sampler = SubsetRandomSampler(training_samples)
    trainset = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

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
        
def log_fold(
    model: torch.nn.Module, 
    fold: int, 
    metrics: Mapping[str, float],
    *args, 
    logger: ILogger = Logger(), 
    tracker: ITracker = Tracker(),
    **kwargs):
    try:
        saved_path = saver.save(model, mode="fold_eval")
        model_params_path = saved_path.absolute()
        tracker.track(metrics, model.__class__.__name__, model_params_path, *args, fold=fold, **kwargs)
    except Exception as ex:
        logger.log(f"An error occured when computing metrics or storing model: {traceback.format_exc()}")

def kfoldcv(
    model_provider: IModelProvider,
    dataset: ITensorAudioDataset, 
    metric_computer: IMetricComputer,
    dataset_verifier: IDatasetVerifier,
    device: str,
    from_checkpoint: Union[bool, pathlib.PosixPath] = False,
    kfolds: int = 5, 
    batch_size: int = 8, 
    num_workers: int = 1, 
    train_kwargs: Mapping = {}, 
    tracker_kwargs: Mapping = {},
    folder_ref: type = KFold,
    logger: ILogger = Logger()):
    
    verify_arguments(
        model_provider=model_provider,
        dataset=dataset,
        dataset_verifier=dataset_verifier,
        device=device,
        kfolds=kfolds,
        batch_size=batch_size,
        num_workers=num_workers
    )

    started_at = datetime.now()
    folds = folder_ref(n_splits=kfolds, shuffle=True)
    
    for fold, (training_samples, test_samples) in enumerate(folds.split(dataset)):
        logger.log("----------------------------------------")
        logger.log(f"Start fold {fold}")
        
        model = model_provider.instantiate()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"The return type of the model_provider.instantiate() has invalid type, must be torch.nn.Module instance, but received {type(model)}")

        model.to(device)

        model, optimizer = train(model, dataset, training_samples, from_checkpoint=from_checkpoint, batch_size=batch_size, num_workers=num_workers, **train_kwargs)
        
        truth, predictions = eval(model, dataset, test_samples, batch_size, num_workers, device=device)
        metrics = metric_computer(truth, predictions)
        
        log_fold(
            model, 
            fold, 
            metrics,
            model_provider.properties,
            train_kwargs,
            kfolds=kfolds, 
            device=device, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            started_at=started_at.strftime(config.DATETIME_FORMAT), 
            tracker_kwargs=tracker_kwargs
        )

        logger.log(f"End fold {fold}")
        logger.log("----------------------------------------")
    
    clear_checkpoints(model, optimizer)