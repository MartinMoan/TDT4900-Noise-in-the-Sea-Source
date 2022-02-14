#!/usr/bin/env python3
import pathlib
import sys
import os
from datetime import datetime

import pandas as pd
from rich import print
import numpy as np
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def _create_results_path_if_not_exists(path):
    if path.suffix != "":
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)

def _get_all_dicts(*args):
    return [arg for arg in args if type(arg) == dict]

def combine_args(*args, **kwargs):
    all_dict = {}
    for arg in _get_all_dicts(*args):
        for key in arg.keys():
            all_dict[key] = arg[key]
    for key in kwargs.keys():
        all_dict[key] = kwargs[key]
    return all_dict

def _add_default_columns(df, *args, **kwargs):
    if config.LOGGED_AT_COLUMN not in df.columns:
        df[config.LOGGED_AT_COLUMN] = np.nan
    return df

def _init_missing_df_columns(df, *args, **kwargs):
    arguments = combine_args(*args, **kwargs)
    for col in arguments.keys():
        if col not in df.columns:
            df[col] = np.nan
    df = _add_default_columns(df, *args, **kwargs)
    return df

def _get_dataframe(*args, **kwargs):
    df = None
    if not config.EXPERIMENTS_FILE.exists():
        df = pd.DataFrame(columns=combine_args(*args, **kwargs).keys())
    else:
        df = pd.read_csv(config.EXPERIMENTS_FILE)
        df = _init_missing_df_columns(df, *args, **kwargs)
    return df

def _empty_row(df, *args, **kwargs):
    arguments = combine_args(*args, **kwargs)
    cols = list(set(df.columns).union(set(arguments.keys())))
    return {key: None for key in cols}

def _add_data(df, *args, **kwargs):
    data = combine_args(*args, **kwargs)
    row = _empty_row(df, *args, **kwargs)
    row[config.LOGGED_AT_COLUMN] = datetime.now().strftime(config.DATETIME_FORMAT)
    for key in data.keys():
        row[key] = data[key]
    return df.append(row, ignore_index=True)
    
def _save(df, *args, **kwargs):
    df.to_csv(config.EXPERIMENTS_FILE, index=False)

def _get_git_commit_hash():
    repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
    sha = repo.head.object.hexsha
    repo_url = repo.remote().url
    current_branch = repo.active_branch
    current_commit = repo.head.commit
    out = {"repo": repo_url, "branch": str(current_branch), "commit": str(current_commit)}
    return out
    
def _get_command_and_arguments():
    repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
    repo_path = pathlib.Path(repo.working_dir)
    script = pathlib.Path(sys.argv[0]).absolute().relative_to(repo_path.parent.absolute())
    sys.argv[0] = str(script)
    return {"command": " ".join(sys.argv)}

def _get_host_info():
    uname = os.uname()
    output = {
        "sysname": uname.sysname,
        "nodename": uname.nodename,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine
    }
    return output

def _add_default_arguments(*args, **kwargs):
    args = list(args)
    args.append(_get_command_and_arguments())
    args.append(_get_git_commit_hash())
    args.append(_get_host_info())
    args = tuple(args)
    return args, kwargs

def track(*args, **kwargs):
    args, kwargs = _add_default_arguments(*args, **kwargs)
    _create_results_path_if_not_exists(config.EXPERIMENTS_FILE)
    df = _get_dataframe(*args, **kwargs)
    df = _add_data(df, *args, **kwargs)
    _save(df, *args, **kwargs)

if __name__ == "__main__":
    d = {"akey": "yeah!"}
    track(1, 2, d, astring="Test", something="a string", something_new="Does this show up?")