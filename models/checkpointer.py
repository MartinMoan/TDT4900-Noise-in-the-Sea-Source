#!/usr/bin/env python3
from datetime import datetime, timedelta
import warnings
import sys
import pathlib
import json
from typing import Mapping, Union, Iterable
import re
import shutil

import numpy as np
import torch
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from rich import print

def _create_checkpoint_dirname(started_at: datetime, model: torch.nn.Module, optimizer: torch.nn.Module) -> pathlib.PosixPath:
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    checkpoint_dirname = pathlib.Path(f"model_{model.__class__.__name__}_optimizer_{optimizer.__class__.__name__}_{timestamp}".lower())
    return config.CHECKPOINTS_PATH.joinpath(checkpoint_dirname)

def _comparator(value):
    matches = re.search(r"model_(.*)_optimizer_(.*)_([0-9]{8}_[0-9]{6}_[0-9]{6})", str(value))
    model, optimizer, dt = matches.groups()
    time = datetime.strptime(dt, "%Y%m%d_%H%M%S_%f")
    return time

def _dirs_with_model_optimizer_values(model: torch.nn.Module, optimizer: torch.nn.Module) -> Iterable[pathlib.PosixPath]:
    dirs = [glob for glob in list(config.CHECKPOINTS_PATH.glob("**/*")) if glob.is_dir()]
    filtered = []
    for d in dirs:
        matches = re.search(r"model_(.*)_optimizer_(.*)_([0-9]{8}_[0-9]{6}_[0-9]{6})", str(d))
        modelname, optimizername, dt = matches.groups()
        if modelname == str(model.__class__.__name__).lower() and optimizername == str(optimizer.__class__.__name__).lower():
            filtered.append(d)
    return filtered

def _most_recent_checkpoint(model: torch.nn.Module, optimizer: torch.nn.Module) -> pathlib.PosixPath:
    sorted_dirs = sorted(_dirs_with_model_optimizer_values(model, optimizer), key=_comparator)
    if len(sorted_dirs) > 0:
        return sorted_dirs[0]
    else:
        return None

def load(model: torch.nn.Module, optimizer: torch.nn.Module, local_variables: Mapping[str, any], checkpoint_dir: Union[str, pathlib.PosixPath] = None):
    if checkpoint_dir is None:
        checkpoint_dir = _most_recent_checkpoint(model, optimizer)
        if checkpoint_dir is None:
            warnings.warn(f"No checkpoint directory could be found for model {model.__class__.__name__} and optimizer {optimizer.__class__.__name__}")
            return model, optimizer, local_variables
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
    
    model_params_filename = "model_state_dict.pt"
    optimizer_params_filename = "optimizer_state_dict.pt"
    div_params_filename = "locals.json"

    model_state_dict = torch.load(checkpoint_dir.joinpath(model_params_filename))
    optimizer_state_dict = torch.load(checkpoint_dir.joinpath(optimizer_params_filename))
    
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    with open(checkpoint_dir.joinpath(div_params_filename), "r") as file:
        stored_local_variables = json.load(file)
        for key in stored_local_variables.keys():
            local_variables[key] = stored_local_variables[key]

    return model, optimizer, local_variables

def clear_checkpoints(model: torch.nn.Module, optimizer: torch.nn.Module) -> None:
    dirs = _dirs_with_model_optimizer_values(model, optimizer)
    print("Clearing checkpoints...")
    for directory in dirs:
        print(f"Deleting checkpoint directory {directory}...")
        shutil.rmtree(directory)

def checkpoint(
    started_at: datetime,
    checkpoint_td: timedelta, 
    model: torch.nn.Module, 
    optimizer: torch.nn.Module, 
    local_variables: Mapping[str, any]) -> None:
    """Checkpoint a current training iteration

    Args:
        started_at (datetime): The program execution start time. 
        checkpoint_td (timedelta): The timedelta between checkpoints. Checkpoints will be made with this interval. F.ex.: checkpoint_td = timedelta(minutes=5) will store a checkpoint every 5 minutes. 
        model (torch.nn.Module): The pytorch model to store, will store the model state_dict.
        optimizer (torch.nn.Module): The pytorch optimizer to store, will store the model state_dict.
        local_variables (dict): Return value of locals() 

    Returns:
        datetime: Datetime of most recent checkpoint.
    """
    locals_to_checkpoint = {
        "batch_size": local_variables["batch_size"],
        "num_workers": local_variables["num_workers"],
        "epochs": local_variables["epochs"],
        "epoch": local_variables["epoch"],
        "lr": local_variables["lr"],
        "weight_decay": local_variables["weight_decay"],
        "device": str(local_variables["device"]),
        "lossfunction": str(local_variables["lossfunction"].__class__.__name__),
        "_checkpoint_created_at": str(datetime.now().strftime(config.DATETIME_FORMAT))
    }
    
    import pandas as pd
    if local_variables is None:
        raise Exception("local_variables is None. No parameters to store")
        
    checkpoint_dir = _create_checkpoint_dirname(started_at, model, optimizer)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_params_filename = "model_state_dict.pt"
    optimizer_params_filename = "optimizer_state_dict.pt"
    div_params_filename = "locals.json"

    torch.save(model.state_dict(), checkpoint_dir.joinpath(model_params_filename))
    torch.save(optimizer.state_dict(), checkpoint_dir.joinpath(optimizer_params_filename))
    with open(checkpoint_dir.joinpath(div_params_filename), "w") as file:
        json.dump(locals_to_checkpoint, file)