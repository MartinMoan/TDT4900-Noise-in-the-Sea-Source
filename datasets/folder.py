#!/usr/bin/env python3
import sys
import pathlib
from typing import Iterable, Generator, Tuple, Mapping

import git
from sklearn.model_selection import KFold

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IFolder

class BasicKFolder(IFolder):
    def __init__(self, n_splits: int, shuffle: bool = False, random_state: int = None) -> None:
        super().__init__()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._folder = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    
    def split(self, iterable: Iterable[any]) -> Generator[Tuple[Iterable[int], Iterable[int]], None, None]:
        for index, (train, test) in enumerate(self._folder.split(iterable)):
            yield (train, test)

    @property
    def properties(self) -> Mapping[str, any]:
        return {"n_splits": self.n_splits, "shuffle": self.shuffle, "random_state": self.random_state}