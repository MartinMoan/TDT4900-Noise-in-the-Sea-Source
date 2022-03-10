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
import sheets

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
        if type(kwargs[key]) == dict:
            for key2 in kwargs[key].keys():
                all_dict[key2] = kwargs[key][key2]
        else:
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
    return sheets.load_csv()

def _empty_row(df, *args, **kwargs):
    arguments = combine_args(*args, **kwargs)
    cols = list(set(df.columns).union(set(arguments.keys())))
    return {key: None for key in cols}

def _add_data(df, *args, **kwargs):
    data = combine_args(*args, **kwargs)
    row = _empty_row(df, *args, **kwargs)
    # row[config.LOGGED_AT_COLUMN] = datetime.now().strftime(config.DATETIME_FORMAT)
    for key in data.keys():
        row[key] = data[key]
    row = pd.DataFrame(data=[row])
    # return df.append(row, ignore_index=True)
    return pd.concat([df, row], ignore_index=True)
    
def _save(df, order=[], col_order=[]):
    # df.to_csv(config.EXPERIMENTS_FILE, index=False)
    # print(col_order)
    sheets.save_csv(df, order=order, col_order=col_order)

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
    args = [str(script)] + sys.argv[1:] 
    return {"command": " ".join(args)}

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
    newargs = []
    newargs.append(_get_command_and_arguments())
    newargs.append(_get_git_commit_hash())
    newargs.append(_get_host_info())
    newargs.append({config.LOGGED_AT_COLUMN: datetime.now().strftime(config.DATETIME_FORMAT)})
    newcols = []
    for arg in newargs:
        newcols += arg.keys()
    args = tuple(list(args) + list(newargs))
    return args, kwargs, newcols

def _set_col_order(df, default_cols, col_order):
    if col_order == [] or col_order is None:
        non_default_cols = list(np.sort(list(set(default_cols).union(set(df.columns)) - set(default_cols))))
        col_order = default_cols + non_default_cols
    else:
        missing_cols = list(set(df.columns).union(default_cols) - set(col_order))
        col_order += missing_cols
    return col_order

def track(*args, order=[config.LOGGED_AT_COLUMN], col_order=[], **kwargs):
    args, kwargs, default_cols = _add_default_arguments(*args, **kwargs)
    _create_results_path_if_not_exists(config.EXPERIMENTS_FILE)
    df = _get_dataframe(*args, **kwargs)
    df = _add_data(df, *args, **kwargs)
    col_order = _set_col_order(df, default_cols, col_order)
    
    _save(df, order=order, col_order=col_order)

if __name__ == "__main__":
    print("Beginning to test tracking...")
    import time
    for i in range(5):
        track(order=["created_at"], col_order=["created_at"])
        time.sleep(2)
    print("Tracking test done!")
