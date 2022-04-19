#!/usr/bin/env python3
import abc
import inspect 
import pathlib
import multiprocessing
import re

from rich import print
import pprint
import json

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

        # args_s = "\n".join([pprint.pformat(arg, indent=4) for arg in args])
        # kwargs_s = pprint.pformat(kwargs, indent=4)
        # s = args_s + kwargs_s
        args_s = "\n".join([prettify(arg, indent=4) for arg in args])
        
        kwargs_s = prettify(kwargs, indent=4)
        content = (args_s + "\n" + kwargs_s).strip()
        
        # content = (pprint.pformat(args) + pprint.pformat(kwargs)).strip()
        lines = re.split(r"\n+", content)
        
        for i, line in enumerate(lines):
            print(header, line)

if __name__ == "__main__":
    text = "[bold red]Firstline[/bold red]\nSecondline\n\nBlank line above"
    logger = Logger()
    logger.log(text)