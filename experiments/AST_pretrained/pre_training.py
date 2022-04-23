#!/usr/bin/env python3
from datetime import datetime, timedelta
from email.mime import audio
import sys
import pathlib
from pydantic import FilePath

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import matplotlib.pyplot as plt
import torch
from oauth2client.service_account import ServiceAccountCredentials
from rich import print
import git
REPO_DIR=pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)
sys.path.insert(0, str(REPO_DIR))

import config
from clipping import ClippedDataset, ClippingCacheDecorator
from ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
from ITensorAudioDataset import FileLengthTensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from limiting import DatasetLimiter
from IDatasetBalancer import DatasetBalancer, CahcedDatasetBalancer, BalancedDatasetDecorator

def main():
    # config.HOME_PROJECT_DIR.joinpath("ExampleImages")
    # credentials_path = REPO_DIR.joinpath("credentials.json")
    # creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, config.SCOPES)
    
    gauth = GoogleAuth()
    drive = GoogleDrive()
    id = "11DaczVpIWOV0G6EbJv0HCm7Osqsb7WZs"

    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    nmels = 128
    hop_length = 512

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    clip_dataset = ClippingCacheDecorator(
        clip_nsamples = clip_length_samples, 
        overlap_nsamples = clip_overlap_samples
    )

    # limited_dataset = DatasetLimiter(clip_dataset, limit=42, randomize=False, balanced=True)
    # limited_tensordatataset = FileLengthTensorAudioDataset(
    #     dataset = limited_dataset,
    #     label_accessor=BinaryLabelAccessor(),
    #     feature_accessor=MelSpectrogramFeatureAccessor()
    # )

    tensordataset = FileLengthTensorAudioDataset(
        dataset=clip_dataset,
        label_accessor=BinaryLabelAccessor(),
        feature_accessor=MelSpectrogramFeatureAccessor()
    )
    
    for i in range(len(tensordataset)):
        _, X, Y = tensordataset[i]
        audiodata = tensordataset._dataset[i]
        if set(Y.numpy()) != set([0.0]):
            print(Y)
        
        if i % 50 == 0:
            print(i, len(tensordataset))
        # import librosa
        # samples, sr = librosa.load(audiodata.filepath, sr=None)

if __name__ == "__main__":
    main()