#!/usr/bin/env python3
"""
Audio Spectrogram Transformer finetuning script, using pretrained parameters trained on AudioSet for speech commands.
Finetuning task is input (batch_size, T-second, M-mel-band spectrogram) tensor output (batch_size, 2) multi-label clip-level classsification to detect biophonic and/or anthropogenic sound event presence in the input clips. 
"""

import argparse
import pathlib
import sys
from typing import Type, Union

import torch
from rich import print
import git
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from GLIDER import GLIDER
from clipping import ClippedDataset
from ICustomDataset import ICustomDataset
from audiodata import LabeledAudioData
import trainer
from ITensorAudioDataset import FileLengthTensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor, ITensorAudioDataset
from IMetricComputer import BinaryMetricComputer
from IDatasetBalancer import BalancedKFolder, DatasetBalancer
from ASTWrapper import ASTWrapper

class DatasetLimiter(torch.utils.data.Dataset):
    def __init__(self, dataset: ICustomDataset, limit: Union[float, int], randomize: bool = False, balanced=True) -> None:
        self._dataset = dataset
        self._balanced = balanced
        self._dataset_indeces = self._get_subset_indeces(dataset, limit, randomize)

    def _get_subset_indeces(self, dataset, limit, randomize):
        _num_in_subset = None
        
        if type(limit) == float:
            if limit > 1.0 or limit < 0.0:
                raise ValueError(limit)
            _num_in_subset = int(limit * len(dataset))
        elif type(limit) == int:
            _num_in_subset = limit
        else:
            raise TypeError

        raw_indeces = np.arange(0, len(dataset))
        if randomize: 
            choices = np.random.choice(raw_indeces, _num_in_subset, replace=False, )
            return list(choices)
        else:
            return list(range(0, _num_in_subset))

    def __len__(self):
        return len(self._dataset_indeces)

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError(index)
        raw_index = self._dataset_indeces[index]
        return self._dataset[raw_index]

def init_args():
    parser = argparse.ArgumentParser(description="AST pretrained AudioSet finetuning script")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="training epochs")
    parser.add_argument("-bs", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-nw", "--num-workers", type=int, default=8, help="num workers for dataloaders")
    parser.add_argument("--prediction-threshold", type=float, default=0.5, help="Prediction confidence threshold. Any prediction with a confidence less than this is not considered a correct prediction (only relevant during evaluation, has no effect on training).")
    parser.add_argument("--force-gpu", action="store_true", default=False, help="Force using CUDA cores. If no CUDA cores are available, will raise an exception and halt the program.")
    parser.add_argument("-cd", "--clip-duration-seconds", type=float, default=10.0, help="The clip duration in seconds to use for the glider audio data.")
    parser.add_argument("-co", "--clip-overlap-seconds", type=float, default=2.0, help="The clip overlap in seconds. Every clip will overlap the pervious clip with this number of seconds.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Dataloading verbosity. Will log the individual local files loaded if set.")
    parser.add_argument("-cp", "--checkpoint", type=str, default=None, help="Path to the checkpoint directory to load model, optimizer and local variables from.")
    parser.add_argument("--from-checkpoint", action="store_true", default=False, help="Wheter to load model, optimizer and local variables from checkpoint. If -cp (--checkpoint) argument is provided, will use that value as the path to load from")
    return parser.parse_args()

def load_model():
    pass

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    nmels = 128
    hop_length = 512

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    clip_dataset = ClippedDataset(
        clip_nsamples = clip_length_samples, 
        overlap_nsamples = clip_overlap_samples
    )

    lr                  =   args.learning_rate
    weight_decay        =   args.weight_decay
    epochs              =   args.epochs
    batch_size          =   args.batch_size
    num_workers         =   args.num_workers
    n_model_outputs     =   2
    # output_activation   =   torch.nn.Sigmoid() # Task is binary classification (audiofile contains biophonic event or not), therefore sigmoid [0, 1]
    loss_ref            =   torch.nn.BCEWithLogitsLoss # BCEWithLogitsLoss combines Sigmoid and BCELoss in single layer/class for more numerical stability. No need for activation function for last layer
    optimizer_ref       =   torch.optim.Adamax

    model = ASTWrapper(
        activation_func = None, # Because loss function is BCEWithLogitsLoss which includes sigmoid activation.
        label_dim=n_model_outputs,
        fstride=10, 
        tstride=10, 
        input_fdim=nmels, 
        input_tdim=n_time_frames, 
        imagenet_pretrain=True, 
        audioset_pretrain=True, 
        model_size='base384',
        verbose=True
    )

    model.freeze_pretrained_parameters()

    model = torch.nn.DataParallel(model)

    metrics_computer    =   BinaryMetricComputer(clip_dataset.classes())

    train_kwargs        =   {"lr": lr, "weight_decay": weight_decay, "epochs": epochs, "loss_ref": loss_ref, "optimizer_ref": optimizer_ref, "device": device}

    from_checkpoint     =   args.checkpoint if args.checkpoint is not None else args.from_checkpoint

    if args.force_gpu and not torch.cuda.is_available():
        raise Exception(f"Force_gpu argument was set, but no CUDA device was found/available. Found device {device}")

    sampling_rate = 128000
    clip_dur_sec = clip_length_samples / sampling_rate
    clip_overlap_sec = clip_overlap_samples / sampling_rate

    print(f"Using device: {device}")

    limited_dataset = DatasetLimiter(clip_dataset, limit=42, randomize=True, )
    print(len(limited_dataset))
    for i in range(len(limited_dataset)):
        print(limited_dataset[i])
    exit()

    dataset = FileLengthTensorAudioDataset(
        dataset=clip_dataset, 
        label_accessor=BinaryLabelAccessor(), 
        feature_accessor=MelSpectrogramFeatureAccessor()
    )

    trainer.kfoldcv(
        model, 
        None,
        dataset,
        metrics_computer,
        device,
        from_checkpoint=from_checkpoint,
        batch_size=batch_size, 
        num_workers=num_workers,
        train_kwargs=train_kwargs, 
        tracker_kwargs={"description": f"AST pretrained ImageNet and AudioSet pretrained model finetuning. Using {clip_dur_sec:.4f} second clips with {clip_overlap_sec:.4f} second overlaps"},
        folder_ref=BalancedKFolder
    )

def main():
    args = init_args()
    train(args)

if __name__ == "__main__":
    main()