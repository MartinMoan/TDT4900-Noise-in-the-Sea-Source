#!/usr/bin/env python3
import abc
import inspect 
import pathlib
import multiprocessing
import re

from rich import print

class ILogger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log(self, *args, **kwargs) -> None:
        raise NotImplementedError

class Logger(ILogger):
    def __init__(self) -> None:
        pass

    def log(self, *args, **kwargs):
        tag = ""
        stack = inspect.stack()
        # print(stack)
        
        caller_locals = stack[1][0].f_locals
        if "self" in caller_locals:
            tag = caller_locals["self"].__class__.__name__
        else:
            if "__name__" in caller_locals:
                caller_filepath = caller_locals["__name__"]
            else:
                caller_filepath = stack[1].filename
            tag = pathlib.Path(caller_filepath).name
        
        proc = multiprocessing.current_process()

        caller = stack[1]
        
        lineno = caller.frame.f_lineno # caller.frame.f_code.co_firstlineno
        caller_name_path = caller.frame.f_code.co_filename
        caller_name = pathlib.Path(caller_name_path).name

        tagstyle_start = "[bold purple]"
        tagstyle_end = "[/bold purple]"

        header = f"[ {tagstyle_start}{tag}{tagstyle_end} PID {proc.pid} {caller_name}:{lineno} ]"
        header_spacing = 60 - (len(header) - (len(tagstyle_start) + len(tagstyle_end)))
        spacer = "".join([" " for i in range(header_spacing)])
        header = f"{header}{spacer}:"

        content = (str(*args) + str(**kwargs)).strip()
        lines = re.split(r"\n+", content)
        
        for i, line in enumerate(lines):
            print(header, line.strip())

if __name__ == "__main__":
    text = "[bold red]Firstline[/bold red]\nSecondline\n\nBlank line above"
    logger = Logger()
    logger.log(text)