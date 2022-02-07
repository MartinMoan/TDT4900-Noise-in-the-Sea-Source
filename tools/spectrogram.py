#!/usr/bin/env python3
import pathlib
import argparse
from re import S
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

_AUDIOFILE_EXTENSIONS = [".wav", ".flac", ".mp3"]

def _float_in_range(min, max):
    def in_range(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} is not a floating-point literal")
        
        if x < min or x > max:
            raise argparse.ArgumentTypeError(f"{x} is not in range [{min}, {max}]")
        return x
    return in_range

def _positive_integer(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not an integer literal")
    
    if x < 0:
        raise argparse.ArgumentTypeError(f"{x} is less than 0, must be a positive integer")
    return x

def _init_args():
    parser = argparse.ArgumentParser(description="A small script to display a spectrogram of an audiofile")
    parser.add_argument("filepath", type=lambda path: pathlib.Path(path).resolve())
    parser.add_argument("-sc", "--scale", type=str, choices=["standard", "mel"], default="standard", help="The y-axis scale. If 'mel' the mel-spectrogram will be displayed. If 'standard' the frequency spectrogram will be displayed.")
    parser.add_argument("-minf", "--min-frequency", type=_float_in_range(0, np.inf), default=0, help="The minimum frequency/mel value to display the spectogram from. Frequency/mel values below this value will not be displayed.")
    parser.add_argument("-maxf", "--max-frequency", type=_float_in_range(0, np.inf), default=np.inf, help="The maximum frequency/mel value to display the spectogram to. Frequency/mel values above this value will not be displayed.")
    parser.add_argument("-s", "--start", type=_float_in_range(0, np.inf), default=0, help="The relative starting time (in seconds) of the recording from which the spectrogram will be displayed.")
    parser.add_argument("-e", "--end", type=_float_in_range(0, np.inf), default=np.inf, help="The relative ending time (in seconds) of the recording to which the spectrogram will be displayed.")
    parser.add_argument("--nfft", type=_positive_integer, default=1025)
    parser.add_argument("--hop_length", type=_positive_integer, default=512)
    parser.add_argument("--nmels", type=_positive_integer, default=128)
    args = parser.parse_args()
    _verify_arguments(args)
    return args

def _verify_arguments(args):
    # Verify that file exists and is audiofile
    if not args.filepath.exists():
        raise Exception(f"The path {str(args.filepath)} does not exist.")
    
    if not args.filepath.is_file():
        raise Exception(f"The path {str(args.filepath)} is not a file.")
    
    if args.filepath.suffix not in _AUDIOFILE_EXTENSIONS:
        raise Exception(f"The path {str(args.filepath)} has file extension not recognized as an audiofile. Expected {', '.join(_AUDIOFILE_EXTENSIONS)} but received {args.filepath.suffix}")

    if args.start >= args.end:
        raise Exception(f"The starting and ending times arguments are invalid, with the starting timestep being after the ending timestep. Starttime: {args.start} Endtime: {args.end}")
    if args.min_frequency >= args.max_frequency:
        raise Exception(f"The min- and max frequency arguments are invalid, with the min frequency being greater or equal to the max frequency. Min frequency: {args.min_frequency} Max frequency: {args.max_frequency}")

def _time_select(samples, sr, args):
    start_idx = 0
    if args.start != 0:
        start_idx = int(args.start * sr)

    end_idx = len(samples)
    if args.end != np.inf:
        end_idx = int(args.end * sr)
    return samples[start_idx:end_idx]

def _extent(samples, sr, args):
    maxf = args.max_frequency if args.max_frequency != np.inf else int(sr / 2)
    minf = args.min_frequency if args.min_frequency != np.inf else 0
    start = args.start if args.start != np.inf else 0
    end = args.end if args.end != np.inf else len(samples) / sr
    extent = [start, end, minf, maxf]
    return extent

def _standard(samples, sr, args):
    freqs, times, values = signal.spectrogram(samples, fs=sr, nfft=args.nfft)
    freq_idxs = np.where((freqs >= args.min_frequency) & (freqs <= args.max_frequency))
    values = values[freq_idxs, :][0]
    values = np.flip(librosa.power_to_db(values), axis=0)

    plt.imshow(values, aspect="auto", extent=_extent(samples, sr, args))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()

def _melspect(samples, sr, args):
    start, end, fmin, fmax = _extent(samples, sr, args)
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(samples, n_fft=args.nfft, hop_length=args.hop_length, sr=sr, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [mel]")
    # ax.set_xticks(np.linspace(start, end, 10))
    plt.show()


def spectrogram(args):
    # samples, sr = librosa.load(args.filepath, sr=None)
    samples, sr = librosa.load(librosa.ex("trumpet"))
    samples = _time_select(samples, sr, args)
    if args.scale == "standard":
        _standard(samples, sr, args)
    else:
        _melspect(samples, sr, args)
    
def main():
    args = _init_args()
    spectrogram(args)

if __name__ == "__main__":
    main()