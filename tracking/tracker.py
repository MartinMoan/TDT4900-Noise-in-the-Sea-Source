#!/usr/bin/env python3
import pathlib
import sys
import os
from datetime import datetime
from typing import Mapping, Union, Iterable

import pandas as pd
from rich import print
import numpy as np
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
import sheets

def _get_all_dicts(*args):
    return [arg for arg in args if type(arg) == dict]

def _combine_args_and_kwargs(*args: Iterable, **kwargs: Mapping[str, any]) -> Mapping[str, any]:
    all_dict = {}
    for arg in _get_all_dicts(*args):
        for key in arg.keys():
            all_dict[key] = arg[key]
    flattened_args = pd.json_normalize(all_dict).to_dict(orient="records")[0]
    flattened_kwargs = pd.json_normalize(kwargs).to_dict(orient="records")[0]
    combined_dict = flattened_args
    for key in flattened_kwargs.keys():
        if key not in combined_dict.keys():
            combined_dict[key] = flattened_kwargs[key]
        else:
            newkey = f"kwargs.{key}"
            combined_dict[newkey] = flattened_kwargs[key]
    return combined_dict

def _empty_row(df: pd.DataFrame, data: Mapping[str, any]) -> Mapping[str, any]:
    cols = list(set(df.columns).union(set(data.keys())))
    return {key: None for key in cols}

def _add_data(df, data):
    row = _empty_row(df, data)
    row = pd.DataFrame(data=[row])
    return pd.concat([df, row], ignore_index=True)

def _default_values():
    repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
    repo_path = pathlib.Path(repo.working_dir)
    repo_url = repo.remote().url
    current_branch = repo.active_branch
    current_commit = repo.head.commit
    script = pathlib.Path(sys.argv[0]).absolute().relative_to(repo_path.parent.absolute())
    cmd = [str(script)] + sys.argv[1:] 

    uname = os.uname()
    output = {
        "sysname": uname.sysname,
        "nodename": uname.nodename,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "command": " ".join(cmd),
        "repo": repo_url, 
        "branch": str(current_branch), 
        "commit": str(current_commit)
    }
    return output

def _add_required_values(data: Mapping[str, any], metrics: Mapping[str, float], model: str, model_parameters_path: Union[str, pathlib.PosixPath]):
    data[config.LOGGED_AT_COLUMN] = datetime.now().strftime(config.DATETIME_FORMAT)
    data["model"] = model
    data["model_parameters_path"] = model_parameters_path

    flattened_metrics = pd.json_normalize(metrics).to_dict(orient="records")[0]
    for key in flattened_metrics.keys():
        data[key] = flattened_metrics[key]
    return data

def _add_optional_values(data: Mapping[str, any], *args: Iterable, **kwargs: Mapping[str, any]) -> Mapping[str, any]:
    new_values = _combine_args_and_kwargs(*args, **kwargs)
    for key in new_values.keys():
        data[key] = new_values[key]
    return data

def track(metrics: Mapping[str, float], model: str, model_parameters_path: Union[str, pathlib.PosixPath], *args, **kwargs):
    data = _default_values()
    data = _add_required_values(data, metrics, model, model_parameters_path)
    col_order = [config.LOGGED_AT_COLUMN] + list(data.keys())
    data = _add_optional_values(data, *args, **kwargs)

    df = sheets.load_csv()
    df = _add_data(df, data)

    sheets.save_csv(df, order=[config.LOGGED_AT_COLUMN], col_order=col_order)

if __name__ == "__main__":
    print("Beginning to test tracking...")
    import time
    for i in range(5):
        track(order=["created_at"], col_order=["created_at"])
        time.sleep(2)
    print("Tracking test done!")
