#!/usr/bin/env python3
import pathlib
import argparse

import simpleaudio as sa
import numpy as np
import librosa
from scipy import signal

_AUDIOFILE_EXTENSIONS = [".wav", ".flac", ".mp3"]
def _positive_integer(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not an integer literal")
    
    if x < 0:
        raise argparse.ArgumentTypeError(f"{x} is less than 0, must be a positive integer")
    return x

def init_args():
    parser = argparse.ArgumentParser("Simple script to play audio wit")
    parser.add_argument("filepath", type=lambda path: pathlib.Path(path).resolve())
    parser.add_argument("-s", "--start", type=_positive_integer, default=0, help="Start time of recording to play")
    parser.add_argument("-e", "--end", type=_positive_integer, default=np.inf, help="End time of recording to play")
    args = parser.parse_args()
    _verify_args(args)
    return args

def _verify_args(args):
    if args.filepath.suffix not in _AUDIOFILE_EXTENSIONS:
        raise Exception(f"The path {str(args.filepath)} has file extension not recognized as an audiofile. Expected {', '.join(_AUDIOFILE_EXTENSIONS)} but received {args.filepath.suffix}")

def load_samples(args):
    start, end = args.start, args.end
    dur = None
    if args.end != np.inf:
        dur = end - start
    
    return librosa.load(args.filepath, sr=None, mono=True, offset=start, duration=dur)

def _select_timewindow(samples, sr, args):
    start, end = args.start, args.end
    if end == np.inf:
        end = len(samples) * sr
    start_idx, end_idx = sr * start, sr * end
    return samples[start_idx:end_idx]

def _resample(samples, sr):
    new_sr = 44100
    N = len(samples)
    duration_sec = N / sr
    new_N = int(new_sr * duration_sec)
    samples = signal.resample(samples, new_N)
    return samples, new_sr

def main():
    args = init_args()
    
    samples, sr = load_samples(args)
    samples, sr = _resample(samples, sr)
    # samples = samples * 2
    print(np.min(samples), np.max(samples))
    
    audio = samples * (2**15 - 1) / np.max(np.abs(samples))
    audio = samples.astype(np.int16)
    num_channels = 1
    bytes_per_sample = 2
    print(audio)
    print(np.min(audio), np.max(audio))
    print("Playgin audio...")
    player = sa.play_buffer(audio, num_channels, bytes_per_sample, sr)
    print(player.is_playing())
    player.wait_done()
    print(player.is_playing())
    print("done")


if __name__ == "__main__":
    main()