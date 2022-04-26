#!/usr/bin/env python3
from http import client
import multiprocessing
from datetime import datetime
import sys
import pathlib
from typing import Mapping, Iterable, Tuple

import git
import pandas as pd

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ITabularLogger
from tracking.sheets import SheetClient

class TabularLogger(ITabularLogger):
    def __init__(self) -> None:
        super().__init__()

        results_path = config.HOME_PROJECT_DIR.joinpath("results")
        
        datetime_format = "%Y-%m-%dT%H%M%S_%f"
        self.filename = f"{datetime.now().strftime(datetime_format)}.results"
        
        if not results_path.exists():
            results_path.mkdir(parents=False, exist_ok=False)

        self.filepath = results_path.joinpath(self.filename)

    def add_row(self, row: Mapping[str, any]) -> None:
        flattened_row = pd.json_normalize(row).to_dict(orient="records")[0]
        with open(self.filepath, "a") as file:
            file.write(f"{str(flattened_row)}\n\n")

    def format(self, order_by: Iterable[Tuple[str, str]] = ..., col_order: Iterable[str] = ...) -> None:
        return
        
