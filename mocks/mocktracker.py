#!/usr/bin/env python3
import pathlib
import sys
from typing import Mapping

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ITracker, ILoggerFactory, ITensorAudioDataset

class MockTracker(ITracker):
    def __init__(self, logger_factory: ILoggerFactory) -> None:
        super().__init__()
        self.logger = logger_factory.create_logger()

    def track(self, trackables: Mapping[str, any]) -> None:
        self.logger.log(f"{self.__class__.__name__} tracking: {trackables}")
    
    def track_dataset(self, dataset: ITensorAudioDataset) -> None:
        self.logger.log(f"{self.__class__.__name__} tracking dataset: {dataset.__class__.__name__} with length: {len(dataset)}")