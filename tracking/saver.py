#!/usr/bin/env python3
import sys
import pathlib
from datetime import datetime
import warnings
import traceback

import git
import torch
import sklearn.model_selection

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ISaver, ILoggerFactory

class Saver(ISaver):
    def __init__(self, logger_factory: ILoggerFactory) -> None:
        super().__init__()
        self.logger = logger_factory.create_logger()

    def save(self, model: torch.nn.Module, **kwargs) -> pathlib.Path:
        path = self.filepath(model, **kwargs)
        if config.ENV == "prod":
            try:
                torch.save(model.state_dict(), path)
            except Exception as ex:
                self.logger.log(f"An error occured when computing metrics or storing model: {traceback.format_exc()}")
        else:
            warnings.warn("The current environment is not set to 'prod' so no model parameters will be saved by saver.py for the current session.")        
        return path

    def filepath(self, model, **kwargs):
        now = datetime.now()
        t = now.strftime("%Y%m%d_%H%M%S_%f")
        kvs = ""
        for key in kwargs.keys():
            kvs += f"{str(key)}_"
            if type(kwargs[key]) == float:
                kvs += f"{kwargs[key]:.5f}_"
            elif type(kwargs[key]) == str:
                kvs += f"{kwargs[key]}_"
            
        return config.DEFAULT_PARAMETERS_PATH.joinpath(pathlib.Path(f"{str(model.__class__.__name__).lower()}_{kvs}{t}.pt"))

    # def get_state_dict(path, model_ref, *model_args, **model_kwargs):
    #     params = torch.load(path)
    #     model = model_ref(*model_args, **model_kwargs)
    #     model.load_state_dict(params)
    #     return model