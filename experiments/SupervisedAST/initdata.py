#!/usr/bin/env python3 
# python initdata.py -nmels 128 -hop_length 512 -nfft 1024 -clip_duration_seconds 10.0 -clip_overlap_seconds 4.0
import os
import argparse
import sys
import pathlib

import git
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from tracking.logger import SlurmLogger
from tracking.loggerfactory import LoggerFactory
from datasets.initdata import create_tensorset

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nmels", type=int, required=True)
    parser.add_argument("-nfft", type=int, required=True)
    parser.add_argument("-hop_length", type=int, required=True)
    parser.add_argument("-clip_duration_seconds", type=float, required=True)
    parser.add_argument("-clip_overlap_seconds", type=float, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = getargs()
    proc = os.environ.get("SLURM_PROCID")
    print(f"SLURM_PROCID: {proc}")

    logger_factory = LoggerFactory(logger_type=SlurmLogger)

    tensorset, balancer = create_tensorset(
        logger_factory=logger_factory,
        nfft=args.nfft,
        nmels=args.nmels,
        hop_length=args.hop_length,
        clip_duration_seconds=args.clip_duration_seconds,
        clip_overlap_seconds=args.clip_overlap_seconds
    )