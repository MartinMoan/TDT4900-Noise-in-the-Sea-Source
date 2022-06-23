#!/usr/bin/env python3
import argparse
from datetime import timedelta
import sys
import pathlib

import git
from rich import print
from xgboost import XGBClassifier
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew
labels_path = pathlib.Path("/Users/martinmoan/TDT4900-Noise-in-the-Sea/code/datamodule/metadata/labels.csv")
metadata_path = pathlib.Path("/Users/martinmoan/TDT4900-Noise-in-the-Sea/code/datamodule/metadata/metadata.csv")

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

sns.set()
sns.set_palette("colorblind")

def normalize(a):
    return np.divide(a - np.mean(a), np.std(a))

def scale(a, min=0, max=1):
    return ((max - min) * (a - np.min(a))) / (np.max(a) - np.min(a)) + min

def label(recordings_df, labels_df, positive_val=True, negative_val=False):
    recordings_df[labels_df.source_class.unique()] = negative_val
    for i in range(len(labels_df)):
        label = labels_df.iloc[i]
        overlapping = recordings_df[(recordings_df.start_time <= label.end_time) & (recordings_df.end_time >= label.start_time)].index
        recordings_df.loc[overlapping, label.source_class] = True
    return recordings_df

def to_dt(df):
    df.start_time = pd.to_datetime(df.start_time, errors="coerce")
    df.end_time = pd.to_datetime(df.end_time, errors="coerce")
    return df

def existing(df):
    return df[(df.filepath.apply(lambda filepath: pathlib.Path(filepath).exists()))]

def get_clips(duration=10.0, overlap=4.0):
    labels = to_dt(pd.read_csv(labels_path))
    metadata = to_dt(pd.read_csv(metadata_path))

    data = []
    for i in tqdm(range(len(metadata))):
        recording = metadata.iloc[i]
        recording_duration = recording.duration_seconds
        n_clips = int((recording_duration - overlap) // (duration - overlap))
        for j in range(n_clips):
            offset = (duration - overlap) * j
            start_time = recording.start_time + timedelta(seconds=offset)
            clip = dict(
                start_time=start_time,
                end_time=start_time + timedelta(seconds=duration),
                filepath=recording.filepath,
                sample_rate=recording.sample_rate,
                num_samples=int(recording.sample_rate * duration),
                duration_seconds=duration,
                offset=offset
            )
            data.append(clip)
    return pd.DataFrame(data=data)

def extract_features(filepath, offset=0.0, duration=None):
    samples, sr = librosa.load(filepath, sr=None, offset=offset, duration=duration)
    mfcc = librosa.feature.mfcc(samples, sr=sr, n_mfcc=20)
    zcr = librosa.feature.zero_crossing_rate(samples)
    rolloff = librosa.feature.spectral_rolloff(samples)
    centroid = librosa.feature.spectral_centroid(samples)

    features = [mfcc, zcr, rolloff, centroid]
    transforms = [np.mean, np.std, skew, np.max, np.min]
    data = []
    for f in features:
        for t in transforms:
            data = np.concatenate((data, t(f, axis=f.ndim - 1)))
    
    return data

def main():
    clips = existing(to_dt(pd.read_csv(pathlib.Path(__file__).parent.joinpath("clips.csv"))))
    clips = clips.reset_index(drop=True)
    featuredata = []
    for i in tqdm(range(len(clips))):
        clip = clips.iloc[i]
        features = extract_features(clip.filepath, offset=clip.offset, duration=clip.duration_seconds)
        featuredata.append(features)
    features = pd.DataFrame(data=featuredata).reset_index(drop=True)
    data = pd.concat((clips, features), axis=1)
    data.to_csv(pathlib.Path(__file__).parent.joinpath("features.csv"), index=False)

if __name__ == "__main__":
    df = get_clips()
    df.to_csv(pathlib.Path(__file__).parent.joinpath("clips.csv"), index=False)
    main()


