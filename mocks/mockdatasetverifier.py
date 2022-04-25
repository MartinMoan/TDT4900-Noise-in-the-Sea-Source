#!/usr/bin/env python3
import sys
import pathlib

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ITensorAudioDataset, IDatasetVerifier

class MockDatasetVerifier(IDatasetVerifier):
    def __init__(self) -> None:
        super().__init__()

    def verify(self, dataset: ITensorAudioDataset) -> bool:
        return True