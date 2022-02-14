#!/usr/bin/env python3
import pathlib
import sys
from datetime import datetime

import torch
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def filepath(model, **kwargs):
    now = datetime.now()
    t = now.strftime("%Y%m%d_%H%M%S_%f")
    kvs = ""
    for key in kwargs.keys():
        kvs += f"{str(key)}_"
        if type(kwargs[key]) == float:
            kvs += f"{kwargs[key]:.5f}_"
        
    return config.DEFAULT_PARAMETERS_PATH.joinpath(pathlib.Path(f"{str(model.__class__.__name__).lower()}_{kvs}{t}.pt"))

def save(model, **kwargs):
    path = filepath(model, **kwargs)
    torch.save(model.state_dict(), path)
    return path

def get_state_dict(path, model_ref, *model_args, **model_kwargs):
    params = torch.load(path)
    model = model_ref(*model_args, **model_kwargs)
    model.load_state_dict(params)
    return model

if __name__ == "__main__":
    model = torch.nn.Sequential(torch.nn.Linear(10, 2))
    # save(model, stuff=3, something_else="bad", shouldbefloat=1.2)
    fp =filepath(model, mode="fold_eval", accuracy=0.5, precision=0.6, f1=0.01, roc_auc=0.123)
    print(fp)