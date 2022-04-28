#!/usr/bin/env python3
import gc
from datetime import datetime
import sys
import inspect 
import pathlib
import multiprocessing
import re
from typing import Iterable, Tuple

from rich import print

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ILogger, ILogFormatter

def prettify(arg, indent=4):
    def nested(a, i, recursion_counter=1):
        ind = "".join([" " for i in range(int(i*recursion_counter))])
        smallind = "".join([" " for i in range(int(i*(recursion_counter-1)))])
        if type(a) == dict:
            if len(a) == 0:
                return ""
            s = "{\n"
            for key in a.keys():
                value = nested(a[key], indent, recursion_counter=recursion_counter+1)
                s += f"{ind}{key}: {value}\n"
            s += smallind + "}"
            return s
        else:
            return str(a)

    return nested(arg, indent)

class LogFormatter(ILogFormatter):
    def header(self, *args, with_tagstyle: bool = True, **kwargs) -> str:
        dt = datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")
        dts = f"[ {dt} ]"
        tag = ""
        stack = inspect.stack()
        
        caller_locals = stack[2][0].f_locals
        if "self" in caller_locals:
            tag = caller_locals["self"].__class__.__name__
        else:
            if "__name__" in caller_locals:
                caller_filepath = caller_locals["__name__"]
            else:
                caller_filepath = stack[2].filename
            tag = pathlib.Path(caller_filepath).name
        
        proc = multiprocessing.current_process()

        caller = stack[2]
        
        lineno = caller.frame.f_lineno # caller.frame.f_code.co_firstlineno
        caller_name_path = caller.frame.f_code.co_filename
        caller_name = pathlib.Path(caller_name_path).name

        tagstyle_start, tagstyle_end = "", ""
        if with_tagstyle:
            tagstyle_start = "[bold purple]"
            tagstyle_end = "[/bold purple]"

        header = f"[ {tagstyle_start}{tag}{tagstyle_end} PID {proc.pid} {caller_name}:{lineno} ]"
        header_spacing = 60 - (len(header) - (len(tagstyle_start) + len(tagstyle_end)))
        spacer = "".join([" " for i in range(header_spacing)])
        header = f"{dts}{header}{spacer}:"
        return header

    def format(self, *args, **kwargs) -> Iterable[str]:
        args_s = "\n".join([prettify(arg, indent=4) for arg in args])
        kwargs_s = prettify(kwargs, indent=4)
        content = (args_s + "\n" + kwargs_s).strip()
        lines = re.split(r"\n+", content)
        return lines

class Logger(ILogger):
    def __init__(self, logformatter: ILogFormatter):
        if not isinstance(logformatter, ILogFormatter):
            raise TypeError(f"Arugment logformatter has incorrect type. Expected {ILogFormatter} but received {type(logformatter)}")
        self.logformatter = logformatter
        
        log_dir = config.HOME_PROJECT_DIR.joinpath("logs")
        if not log_dir.exists():
            log_dir.mkdir(parents=False, exist_ok=False)

        datetime_format = "%Y-%m-%dT%H%M%S_%f"
        filename = f"{datetime.now().strftime(datetime_format)}.log"
        self.logfile = log_dir.joinpath(filename)

    def log(self, *args, **kwargs):
        header_with_style = self.logformatter.header(*args, **kwargs)
        basic_header = self.logformatter.header(*args, with_tagstyle=False, **kwargs)
        lines = self.logformatter.format(*args, **kwargs)
        
        with open(self.logfile, "a") as logfile:
            for i, line in enumerate(lines):
                print(header_with_style, line)
                logfile.write(f"{basic_header} {line}\n")

if __name__ == "__main__":
    text = "[bold red]Firstline[/bold red]\nSecondline\n\nBlank line above"
    logger = Logger(logformatter=LogFormatter)
    logger.log(text)