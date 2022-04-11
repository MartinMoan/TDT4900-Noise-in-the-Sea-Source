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
from IDatasetBalancer import DatasetBalancer, BalancerCacheDecorator, BalancedDatasetDecorator

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

    limited_dataset = DatasetLimiter(clip_dataset, limit=42, randomize=False, balanced=True)
    limited_tensordatataset = FileLengthTensorAudioDataset(
        dataset = limited_dataset,
        label_accessor=BinaryLabelAccessor(),
        feature_accessor=MelSpectrogramFeatureAccessor()
    )
    
    for i in range(len(limited_tensordatataset)):
        _, X, Y = limited_tensordatataset[i]
        audiodata = limited_tensordatataset._dataset[i]
        print(audiodata)
        print(audiodata.labels.source_class.unique())
        print(audiodata.labels)
        print(audiodata.samples.shape, len(audiodata.samples) / audiodata.sampling_rate)
        import librosa
        samples, sr = librosa.load(audiodata.filepath, sr=None)
        print(sr)
        
        start_time = datetime(2022, 1, 1, 0,0,0,0)
        a = LabeledAudioData(
            _index = i,
            filepath=audiodata.filepath,
            num_channels=audiodata.num_channels,
            sampling_rate=sr,
            file_start_time=start_time,
            file_end_time=start_time+timedelta(seconds=len(samples)/sr),
            clip_duration=len(samples)/sr,
            clip_offset=0,
            all_labels=audiodata.all_labels,
            labels_dict=audiodata.labels_dict
        )
        feature = MelSpectrogramFeatureAccessor()
        Xfull = feature(a)
        print(Xfull.shape)
        print(Y)
        img = torch.squeeze(X)
        imgfull = torch.squeeze(Xfull)
        plt.imshow(img, aspect="auto")
        plt.show()
        plt.imshow(imgfull, aspect="auto", cmap="viridis")
        plt.show()
        exit()


if __name__ == "__main__":
    main()