#!/usr/bin/env python3
import warnings
from inspect import getframeinfo, stack  
from typing import Union, Type
from rich import print

# source: https://stackoverflow.com/a/287944
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def custom_format(message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, *args, **kwargs):
    # el = getframeinfo(stack()[-1][0])
    # contexts = "".join([c.strip() for c in el.code_context])
    return f"\n{colors.HEADER}{colors.BOLD}{filename}:{lineno}: {category.__name__}:\n\t{colors.WARNING}{message}{colors.ENDC}\n\n"