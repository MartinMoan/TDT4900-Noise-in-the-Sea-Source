#!/usr/bin/env python3
from datetime import timedelta
from inspect import trace
from logging import warning
import pathlib
import getpass
import sys
import os
from types import TracebackType
import warnings
from urllib.parse import unquote

import dotenv
dotenv.load_dotenv()

import git
from rich import print

import CustomWarnings
ENV = os.environ.get("ENV", "dev")
print(ENV)
if ENV == "dev":
    # warnings.warn = CustomWarnings.warn
    warnings.formatwarning = CustomWarnings.custom_format

repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
REPO_DIR = pathlib.Path(repo.working_dir).absolute()

for directory in [d for d in REPO_DIR.glob("**/*") if d.is_dir()]:
    sys.path.insert(0, str(directory))