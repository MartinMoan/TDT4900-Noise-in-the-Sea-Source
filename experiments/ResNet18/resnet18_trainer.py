#!/usr/bin/env python3
import argparse
import pathlib
import sys

import torch
from rich import print
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from GLIDER import GLIDER
from clipping import ClippedDataset
from audiodata import LabeledAudioData
from ResNet18 import ResNet18
import trainer
from ITensorAudioDataset import FileLengthTensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from IMetricComputer import BinaryMetricComputer

def init_args():
    parser = argparse.ArgumentParser(description="ResNet18 training script")
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

def train(args):
    device              =   "cuda" if torch.cuda.is_available() else "cpu"

    clip_duration_sec   =   args.clip_duration_seconds
    clip_overlap_sec    =   args.clip_overlap_seconds
    glider              =   ClippedDataset(clip_duration_seconds = clip_duration_sec, clip_overlap_seconds = clip_overlap_sec)
    dataset             =   FileLengthTensorAudioDataset(dataset=glider, label_accessor = BinaryLabelAccessor(), feature_accessor = MelSpectrogramFeatureAccessor())
    class_information   =   dataset.classes()
    
    lr                  =   args.learning_rate
    weight_decay        =   args.weight_decay
    epochs              =   args.epochs
    batch_size          =   args.batch_size
    num_workers         =   args.num_workers
    n_model_outputs     =   len(class_information.keys())
    # output_activation   =   torch.nn.Sigmoid() # Task is binary classification (audiofile contains biophonic event or not), therefore sigmoid [0, 1]
    loss_ref            =   torch.nn.BCEWithLogitsLoss # BCEWithLogitsLoss combines Sigmoid and BCELoss in single layer/class for more numerical stability. No need for activation function for last layer
    optimizer_ref       =   torch.optim.Adamax

    model_ref           =   ResNet18
    model_kwargs        =   {"n_outputs": n_model_outputs}

    metrics_computer    =   BinaryMetricComputer(glider.classes())

    train_kwargs        =   {"lr": lr, "weight_decay": weight_decay, "epochs": epochs, "loss_ref": loss_ref, "optimizer_ref": optimizer_ref, "device": device}

    from_checkpoint     =   args.checkpoint if args.checkpoint is not None else args.from_checkpoint

    if args.force_gpu and not torch.cuda.is_available():
        raise Exception(f"Force_gpu argument was set, but no CUDA device was found/available. Found device {device}")

    print(f"Using device: {device}")
    trainer.kfoldcv(
        model_ref, 
        model_kwargs, 
        dataset,
        metrics_computer,
        device,
        from_checkpoint=from_checkpoint,
        batch_size=batch_size, 
        num_workers=num_workers,
        train_kwargs=train_kwargs, 
        tracker_kwargs={"description": f"ResNet18 model with {clip_duration_sec} second clip length input and binary class vector output"}
    )

def main():
    args = init_args()
    train(args)

if __name__ == "__main__":
    main()
