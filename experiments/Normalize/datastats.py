#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
from datetime import datetime
import json

import git
from tqdm import tqdm
import torch
import numpy as np
from rich import print

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

import config
from datasets.initdata import create_tensorset
from datasets.tensordataset import TensorAudioDataset

def stats(X, Y):
    data = {
        "min_x_value": np.min(X.numpy()),
        "max_x_value": np.max(X.numpy()),
        "min_y_value": np.min(Y.numpy()),
        "max_y_value": np.max(Y.numpy),
        "mean_x": np.mean(X.numpy()),
        "stddev_x": np.std(X.numpy()),
        "mean_y": np.mean(Y.numpy()),
        "stddev_y": np.std(Y.numpy())
    }
    return data

def main(args):
    tensorset, balancer = create_tensorset(
        nfft=args.nfft,
        nmels=args.nmels,
        hop_length=args.hop_length,
        clip_duration_seconds=args.clip_duration_seconds,
        clip_overlap_seconds=args.clip_overlap_seconds,
    )

    min_x_value = np.inf
    max_x_value = -np.inf
    min_y_value = np.inf
    max_y_value = -np.inf
    mean_xs = []
    stddev_xs = []
    mean_ys = []
    stddev_ys = []

    for i,(X, Y) in enumerate(tensorset):
        if i % 500 == 0:
            print(i, len(tensorset))
        mnx = np.min(X.numpy())
        if mnx < min_x_value:
            min_x_value = mnx
        mxx = np.max(X.numpy())
        if mxx > max_x_value:
            max_x_value = mxx
        mny = np.min(Y.numpy())
        if mny < min_y_value:
            min_y_value = mny
        mxy = np.max(Y.numpy())
        if mxy > max_y_value:
            max_y_value = mxy
        mean_xs.append(np.mean(X.numpy()))
        stddev_xs.append(np.std(X.numpy()))
        mean_ys.append(np.mean(Y.numpy()))
        stddev_ys.append(np.std(Y.numpy()))

    repo = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
    data = {
        "min_x_value": float(min_x_value),
        "max_x_value": float(max_x_value),
        "min_y_value": float(min_y_value),
        "max_y_value": float(max_y_value),
        "mean_x": float(np.mean(mean_xs)),
        "stddev_x": float(np.mean(stddev_xs)),
        "mean_y": float(np.mean(mean_ys)),
        "stddev_y": float(np.mean(stddev_ys)),
        "tensorset_size": len(tensorset),
        "computed_at": datetime.now().strftime(config.DATETIME_FORMAT),
        "commit": str(repo.commit()),
        "branch": str(repo.active_branch),
        "node": os.uname().nodename
    }
    output = pathlib.Path(__file__).parent.joinpath("datastats.json")
    print(f"Storing dataset stats at:\n{str(output.absolute())}")
    print(data)
    with open(output, "w") as file:
        json.dump(data, file)

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nmels", type=int, required=True, help="The number of Mel-bands/filters to use when computing the Mel-spectrogram from the raw audio data. This argument determines the 'vertical' dimensions of the spectrograms.")
    parser.add_argument("-nfft", type=int, required=True, help="The size of window in number of samples to compute the FFT over during the Short-Time Fourier Transform algorithm. A larger window yields better frequency determination, but degrades time determination, and vice-versa.")
    parser.add_argument("-hop_length", type=int, required=True, help="The hop length in number of samples skip between successive windows when computing the Short-Time Fourier Transform of the raw audio. This, together with the '-nfft' argument and the number of samples (determined by sampling rate and '-clip_duration_seconds' argument) in the raw audio data, determines the size of the time-dimension of the resulting spectrograms. If '-hop_length' is equal to '-nfft' successive STFT windows will not overlap, if '-hop_length' < '-nfft' successive STFT windows will overlap by ('-nfft' - '-hop_length' samples)")
    parser.add_argument("-clip_duration_seconds", type=float, required=True, help="The clip duration in seconds to use when clipping the raw audiofiles. Clipping is done 'virtually' before actually reading any samples from the audiofile into memory, by using the known sampling rate and duration of each file. This is done to improve performance and memory efficiency, and such that we can compute the dataset size without first reading all the files, clipping them and aggregating the result.")
    parser.add_argument("-clip_overlap_seconds", type=float, required=True, help="The clip overlap in seconds to use when clipping the audiofiles. Cannot be greater than or equal to the '-clip_duration_seconds' argument.")

    return parser.parse_args()

if __name__ == "__main__":
    args = init()
    main(args)