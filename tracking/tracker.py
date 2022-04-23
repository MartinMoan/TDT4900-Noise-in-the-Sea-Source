#!/usr/bin/env python3
import abc
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
# import config
from globalcontainer import GlobalContainer
from globalconfig import GlobalConfiguration
from sheets import SheetClient
from logger import ILogger

LOGGED_AT_COLUMN = "created_at"
from dependency_injector.wiring import Provide, inject

class ITracker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(
        metrics: Mapping[str, float], 
        model: str, 
        model_parameters_path: Union[str, pathlib.PosixPath], 
        *args,
        **kwargs) -> None:

        raise NotImplementedError

class Tracker(ITracker):
    @inject
    def __init__(self, logger: ILogger = Provide[GlobalContainer.logger], config: GlobalConfiguration = Provide[GlobalContainer.config]) -> None:
        super().__init__()
        self.logger = logger
        self.config = config

    def track(
        self,
        metrics: Mapping[str, float], 
        model: str, 
        model_parameters_path: Union[str, pathlib.PosixPath], 
        *args, 
        order: Iterable[str] = None, 
        col_order: Iterable[str] = None, 
        **kwargs) -> None:

        data = self._default_values()
        data = self._add_required_values(data, metrics, model, model_parameters_path)
        data = self._add_optional_values(data, *args, **kwargs)

        if col_order is None:
            rem = set(data.keys()) - set([self.config.LOGGED_AT_COLUMN])
            col_order = [self.config.LOGGED_AT_COLUMN] + list(rem)
        if order is None:
            order = [self.config.LOGGED_AT_COLUMN]

        self.logger.log(f"Tracking data:\n", data)
        client = SheetClient(logger=self.logger, config = self.config)
        client.add_row(data)
        client.format(order_by=order, col_order=col_order)

    def _get_all_dicts(self, *args):
        return [arg for arg in args if type(arg) == dict]

    def _combine_args_and_kwargs(self, *args: Iterable, **kwargs: Mapping[str, any]) -> Mapping[str, any]:
        all_dict = {}
        for arg in self._get_all_dicts(*args):
            for key in arg.keys():
                all_dict[key] = arg[key]
        
        flattened_args = {} if len(all_dict) == 0 else pd.json_normalize(all_dict).to_dict(orient="records")[0]
        flattened_kwargs = {} if len(kwargs) == 0 else pd.json_normalize(kwargs).to_dict(orient="records")[0]
        
        combined_dict = flattened_args
        for key in flattened_kwargs.keys():
            if key not in combined_dict.keys():
                combined_dict[key] = flattened_kwargs[key]
            else:
                newkey = f"kwargs.{key}"
                combined_dict[newkey] = flattened_kwargs[key]
        return combined_dict

    def _add_data(self, df, data):
        cols = list(set(df.columns).union(set(data.keys())))
        empty_row = {key: None for key in cols}
        for key in data.keys():
            empty_row[key] = data[key]
        df = df.append(empty_row, ignore_index=True)
        return df

    def _default_values(self):
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

    def _add_required_values(self, data: Mapping[str, any], metrics: Mapping[str, float], model: str, model_parameters_path: Union[str, pathlib.PosixPath]):
        data[self.config.LOGGED_AT_COLUMN] = datetime.now().strftime(self.config.DATETIME_FORMAT)
        data["model"] = model
        data["model_parameters_path"] = str(model_parameters_path)

        flattened_metrics = pd.json_normalize(metrics).to_dict(orient="records")[0]
        for key in flattened_metrics.keys():
            data[key] = flattened_metrics[key]
        return data

    def _add_optional_values(self, data: Mapping[str, any], *args: Iterable, **kwargs: Mapping[str, any]) -> Mapping[str, any]:
        new_values = self._combine_args_and_kwargs(*args, **kwargs)
        for key in new_values.keys():
            data[key] = new_values[key]
        return data

if __name__ == "__main__":
    container = GlobalContainer()
    container.init_resources()
    container.wire(modules=[__name__])
    print("Beginning to test tracking...")
    import time
    tracker = Tracker()
    for i in range(5):
        metrics = {
            'accuracy': 0.35,
            'precision': {'Biophonic': 0.0, 'Anthropogenic': 1.0},
            'f1': {'Biophonic': 0.0, 'Anthropogenic': 0.2857142857142857},
            'roc_auc': {'Biophonic': 0.5, 'Anthropogenic': 0.5833333333333334},
            'recall': {'Biophonic': 0.0, 'Anthropogenic': 0.16666666666666666}
        }
        model = "TrackingTestModel"
        model_parameters_path = "/no/parameters/exits.pth"
        tracker.track(metrics, model, model_parameters_path, order=["created_at"], col_order=["created_at"])
        time.sleep(2)
    print("Tracking test done!")
