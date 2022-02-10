#!/usr/bin/env python3
import pathlib
import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal

_AUDIOFILE_EXTENSIONS = [".wav", ".flac", ".mp3"]
_OUTPUT_EXTENSIONS = [".png", ".pgf"]
MPL_PGF_CONFIG = {
    "pgf.texsystem": "pdflatex",
    'pgf.rcfonts': False,
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

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
    parser.add_argument("-fmin", "--min-frequency", type=_float_in_range(0, np.inf), default=0, help="The minimum frequency/mel value to display the spectogram from. Frequency/mel values below this value will not be displayed.")
    parser.add_argument("-fmax", "--max-frequency", type=_float_in_range(0, np.inf), default=np.inf, help="The maximum frequency/mel value to display the spectogram to. Frequency/mel values above this value will not be displayed.")
    parser.add_argument("-s", "--start", type=_float_in_range(0, np.inf), default=0, help="The relative starting time (in seconds) of the recording from which the spectrogram will be displayed.")
    parser.add_argument("-e", "--end", type=_float_in_range(0, np.inf), default=np.inf, help="The relative ending time (in seconds) of the recording to which the spectrogram will be displayed.")
    parser.add_argument("--nfft", type=_positive_integer, default=1025)
    parser.add_argument("--hop_length", type=_positive_integer, default=512)
    parser.add_argument("--nmels", type=_positive_integer, default=128)
    parser.add_argument("--resample", type=_positive_integer, help="New sampling rate to resample data to before computing the spectrogram. Usefull if data has high sampling rate and computing resources are limited.")
    parser.add_argument("--cmap", type=str, choices=plt.colormaps(), default="magma", help="Matplotlib colormap name to use")
    parser.add_argument("-o", "--output", type=lambda path: pathlib.Path(path).resolve())
    parser.add_argument("--no-display", action="store_true", help="Whether to display the figure or not. If this flag is provided the figure will not be displayed but can be stored directly to file. This can be useful if creating figures should be automated, in which case displaying the figure to screen is unnecessary.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Wether to use verbose logging.")
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

    if args.output is not None:
        if args.output.suffix not in _OUTPUT_EXTENSIONS:
            raise Exception(f"The output filename {args.output.name} has invalid extension suffix. Must be one of {', '.join(_OUTPUT_EXTENSIONS)} but received suffix {args.output.suffix}")
        
        if args.output.suffix == ".pgf":
            matplotlib.use("pgf")
            matplotlib.rcParams.update(MPL_PGF_CONFIG)

def get_samples(args):
    samples, sr = librosa.load(args.filepath, sr=None)
    if args.resample is not None:
        N = len(samples)
        duration_sec = N / sr
        new_N = int(args.resample * duration_sec)
        samples = signal.resample(samples, new_N)
        sr = args.resample
    return samples, sr

def _time_select(samples, sr, args):
    start_idx = 0
    if args.start != 0:
        start_idx = int(args.start * sr)

    end_idx = len(samples)
    if args.end != np.inf:
        end_idx = int(args.end * sr)
    return samples[start_idx:end_idx]

def _extent(samples, sr, args):
    maxf = args.max_frequency if args.max_frequency != np.inf else sr / 2
    if args.scale == "mel":
        maxf = args.max_frequency if args.max_frequency != np.inf else hz_to_mel(sr / 2)
    
    minf = args.min_frequency if args.min_frequency != np.inf else 0
    start = args.start if args.start != np.inf else 0
    end = args.end if args.end != np.inf else len(samples) / sr
    extent = [start, end, minf, maxf]
    return extent

def _standard(samples, sr, args):
    _logmessage(args, "Computing the Herz-scale spectrogram...")
    freqs, times, values = signal.spectrogram(samples, fs=sr, nfft=args.nfft)
    values = librosa.power_to_db(values)    
    _logmessage(args, "Done!")
    freq_idxs = np.where((freqs >= args.min_frequency) & (freqs <= args.max_frequency))
    values = values[freq_idxs, :][0]

    plt.imshow(values, aspect="auto", extent=_extent(samples, sr, args), origin="lower", cmap=args.cmap)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")

def hz_to_mel(hz):
    return 2595.0 * np.log10(1 + (hz / 700))

def _melspect(samples, sr, args):
    _logmessage(args, "Computing the mel-spectrogram...")
    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=args.nmels, n_fft=args.nfft))
    _logmessage(args, "Done!")
    start,end,fmin,fmax = _extent(samples, sr, args)
    mels = np.array([hz_to_mel(hz) for hz in np.linspace(0, sr / 2, args.nmels)])
    mel_idxs = np.where((mels >= fmin) & (mels <= fmax))
    S_db = S_db[mel_idxs, :][0]
    plt.imshow(S_db, aspect='auto', extent=[start, end, fmin, min([fmax, hz_to_mel(sr / 2)])], origin="lower", cmap=args.cmap)
    
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [mel]")

def _save(args):
    _logmessage(args, "Storing the figure to file at {args.output}")
    plt.savefig(args.output, bbox_inches="tight")

def _logmessage(namespace_args, *args, **kwargs):
    if namespace_args.verbose:
        print(*args, **kwargs)

def spectrogram(args):
    _logmessage(args, "Loading file samples...")
    samples, sr = get_samples(args)
    _logmessage(args, "File loaded!")
    _logmessage(args, "Selecting required timeframe...")
    samples = _time_select(samples, sr, args)
    _logmessage(args, "Done!")
    _logmessage(args, "Creating the spectrogram...")
    if args.scale == "standard":
        _standard(samples, sr, args)
    else:
        _melspect(samples, sr, args)
    
def main():
    args = _init_args()
    spectrogram(args)
    if args.output:
        _save(args)

    if not args.no_display:
        _logmessage(args, "Displaying the spectrogram...")
        plt.show()

if __name__ == "__main__":
    main()