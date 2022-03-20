#!/usr/bin/env python3
import sys
import pathlib
from datetime import datetime, timedelta
import traceback
import warnings
from typing import Iterable, Mapping, Type

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
import tracker
from ITensorAudioDataset import ITensorAudioDataset
from IMetricComputer import IMetricComputer

def _verify_arguments(
    model_ref: torch.nn.Module, 
    model_kwargs: Mapping, 
    dataset: ITensorAudioDataset, 
    device: str, 
    kfolds: int = 5, 
    batch_size: int = 8, 
    num_workers: int = 1, 
    train_kwargs: Mapping = {}, 
    eval_kwargs: Mapping = {}, 
    tracker_kwargs: Mapping = {}):
    model = model_ref(**model_kwargs)
    
    if not isinstance(model, torch.nn.Module):
        raise Exception("Argument model_ref does not point to the initializer (__init__) method of a torch.nn.Module")
    
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

# def evaluate(model: torch.nn.Module, testset: torch.utils.data.DataLoader, threshold: int = 0.5, device: str = None):
def infer(
    model: torch.nn.Module, 
    dataset: ITensorAudioDataset, 
    test_samples, 
    batch_size: int, 
    num_workers: int, 
    device: str = None):
    
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
    loss_ref: Type[torch.nn.Module] = torch.nn.BCELoss, 
    optimizer_ref: Type[torch.nn.Module] = torch.optim.Adamax, 
    device: str = None,
    checkpoint_dt: timedelta = timedelta(minutes=5)):
    
    _LAST_CHECKPOINT_CREATED_AT = None

    sampler = SubsetRandomSampler(training_samples)
    trainset = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    if config.ENV == "dev":
        for epoch in range(epochs):
            for batch in range(len(trainset)):
                print(f"\t\ttraining epoch {epoch} batch {batch} / {len(trainset)} loss -1 (simulated loss)")
        return model

    optimizer = optimizer_ref(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfunction = loss_ref()

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
            if _LAST_CHECKPOINT_CREATED_AT is None or (datetime.now() - _LAST_CHECKPOINT_CREATED_AT) >= checkpoint_dt:
                locals_list = ["batch_size", "num_workers", "epocs", "lr", "weight_decay", "device"]
                l = locals()
                locals_to_checkpoint = {key: l[key] for key in l if key in locals_list}
                # checkpoint(model, optimizer, **locals_to_checkpoint)
                _LAST_CHECKPOINT_CREATED_AT = datetime.now()

            
            if np.isnan(current_loss):
                raise Exception("Loss is nan...")
            
            epoch_loss += current_loss
            
            print(f"\t\ttraining epoch {epoch} batch {batch} / {len(trainset)} loss {current_loss}")
            
        average_epoch_loss = epoch_loss / len(trainset)
        saver.save(model, mode="training", avg_epoch_loss=average_epoch_loss)
    return model

def checkpoint(model, optimizer, **kwargs):
    import pandas as pd
    if kwargs is not None:
        flattened_kwargs = pd.json_normalize(kwargs, sep=".").to_dict(orient="records")[0]
        state = {"model.state_dict": model.state_dict(), "optimizer.state_dict": optimizer.state_dict(), **flattened_kwargs}
        print(state.keys())
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{model.__class__.__name__}_{now}.pth".lower()
        filepath = config.DEFAULT_PARAMETERS_PATH.joinpath(filename)
        torch.save(state, filepath)

def log_fold(
    model: torch.nn.Module, 
    fold: int, 
    metrics: Mapping[str, float],
    *args, **kwargs):
    try:
        saved_path = saver.save(model, mode="fold_eval")
        model_params_path = saved_path.absolute().relative_to(config.REPO_DIR)
        tracker.track(metrics, model.__class__.__name__, model_params_path, *args, fold=fold, **kwargs)
    except Exception as ex:
        print(f"An error occured when computing metrics or storing model: {traceback.format_exc()}")

def kfoldcv(
    model_ref: Type[torch.nn.Module], 
    model_kwargs: Mapping, 
    dataset: ITensorAudioDataset, 
    metric_computer: IMetricComputer,
    device: str,
    kfolds: int = 5, 
    batch_size: int = 8, 
    num_workers: int = 1, 
    train_kwargs: Mapping = {}, 
    tracker_kwargs: Mapping = {}):
    started_at = datetime.now()
    folds = KFold(n_splits=kfolds, shuffle=True)
    
    for fold, (training_samples, test_samples) in enumerate(folds.split(dataset)):
        print("----------------------------------------")
        print(f"Start fold {fold}")
        print()
        model = model_ref(**model_kwargs)

        model.to(device)

        model = train(model, dataset, training_samples, batch_size=batch_size, num_workers=num_workers, **train_kwargs)
        
        truth, predictions = infer(model, dataset, test_samples, batch_size, num_workers, device=device)
        metrics = metric_computer(truth, predictions)
        
        log_fold(
            model, 
            fold, 
            metrics,
            model_kwargs, 
            train_kwargs,
            kfolds=kfolds, 
            device=device, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            started_at=started_at.strftime(config.DATETIME_FORMAT), 
            tracker_kwargs=tracker_kwargs
        )

        print(f"End fold {fold}")
        print("----------------------------------------")
