#!/usr/bin/env python3
import gc
import inspect
import sys
import pathlib
from typing import Type, Tuple, Mapping, Callable

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ILoggerFactory, ILogger
from tracking.logger import Logger

class LoggerFactory(ILoggerFactory):
    def __init__(self, logger_type: Type = Logger, logger_args: Tuple[any] = (), logger_kwargs: Mapping[str, any] = {}) -> None:
        super().__init__()
        self._logger_type = logger_type
        self._logger_args = logger_args
        self._logger_kwargs = logger_kwargs

    def create_logger(self) -> ILogger:
        loggers = [logger for logger in gc.get_objects() if isinstance(logger, ILogger)]
        if len(loggers) == 0:
            return self._logger_type(*self._logger_args, **self._logger_kwargs)
        return loggers[0]

    def logger_type(self) -> Type[ILogger]:
        return self._logger_type

if __name__ == "__main__":
    factory = LoggerFactory(logger_type = Logger)
    logger = factory.create_logger()
    logger.log("Something")
    newlogger = factory.create_logger()
    newlogger.log("Else")

    another_factory = LoggerFactory(logger_type=Logger)
    another_logger = another_factory.create_logger()
    another_logger.log("Another one!")

    print(another_logger == newlogger, newlogger == logger)