#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
from datetime import datetime
import pickle

from rich import print
import git
import pytorch_lightning as pl

GIT_REPO = git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True)
sys.path.insert(0, str(pathlib.Path(GIT_REPO.working_dir)))
import config
from experiments.SupervisedAST.model import AstLightningWrapper
from datasets.datamodule import ClippedGliderDataModule

def main(hyperparams):
    started = datetime.now()

    pl.seed_everything(hyperparams.seed_value)
    print("Received hyperparams:", vars(hyperparams))

    dataset = ClippedGliderDataModule(
        batch_size=hyperparams.batch_size,
        nfft=hyperparams.nfft,
        nmels=hyperparams.nmels,
        hop_length=hyperparams.hop_length,
        clip_duration_seconds=hyperparams.clip_duration_seconds,
        clip_overlap_seconds=hyperparams.clip_overlap_seconds,
        num_workers=hyperparams.num_workers
    )
    dataset.setup()
    model = AstLightningWrapper.load_from_checkpoint(checkpoint_path=str(hyperparams.checkpoint.absolute()))

    test = dataset.test_dataloader()
    data = []
    for i, (X, Y) in enumerate(test):
        print(i, len(test))
        Yhat = model.forward(X)
        data.append(
            dict(
                spectrograms=X,
                targets=Y,
                preds=Yhat,
                audiodata=test.audiodata(i)
            )
        )
        if i > 10:
            break
    to_store = dict(
        metadata=dict(
            started_at=started,
            **vars(hyperparams),
            **{key: value for key, value in os.environ.items() if "SLURM" in key},
            commit=GIT_REPO.head.commit
        ),
        data=data
    )
    output_filename = f"predictions_{started.strftime(config.DATETIME_FORMAT)}"
    output = hyperparams.output_dir.joinpath(output_filename)
    print(f"Pickling predictions and audiofile information to: {output.absolute()}")
    with open(output, "wb") as file:
        pickle.dump(to_store, file)

def init():
    parser = argparse.ArgumentParser(
        description="Training script to perform supervised training of Audio Spectrogram Transformer (AST) on the GLIDER dataset."
    )
    parser.add_argument("checkpoint", type=pathlib.Path, help="Path to a pytorch lightning .ckpt file")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path.cwd(), help="Directory of output predictions binary file")
    parser.add_argument("-batch_size", type=int, required=True, help="The batch size to use during training, testing and evaluation")
    parser.add_argument("-nmels", type=int, required=True, help="The number of Mel-bands/filters to use when computing the Mel-spectrogram from the raw audio data. This argument determines the 'vertical' dimensions of the spectrograms.")
    parser.add_argument("-nfft", type=int, required=True, help="The size of window in number of samples to compute the FFT over during the Short-Time Fourier Transform algorithm. A larger window yields better frequency determination, but degrades time determination, and vice-versa.")
    parser.add_argument("-hop_length", type=int, required=True, help="The hop length in number of samples skip between successive windows when computing the Short-Time Fourier Transform of the raw audio. This, together with the '-nfft' argument and the number of samples (determined by sampling rate and '-clip_duration_seconds' argument) in the raw audio data, determines the size of the time-dimension of the resulting spectrograms. If '-hop_length' is equal to '-nfft' successive STFT windows will not overlap, if '-hop_length' < '-nfft' successive STFT windows will overlap by ('-nfft' - '-hop_length' samples)")
    parser.add_argument("-clip_duration_seconds", type=float, required=True, help="The clip duration in seconds to use when clipping the raw audiofiles. Clipping is done 'virtually' before actually reading any samples from the audiofile into memory, by using the known sampling rate and duration of each file. This is done to improve performance and memory efficiency, and such that we can compute the dataset size without first reading all the files, clipping them and aggregating the result.")
    parser.add_argument("-clip_overlap_seconds", type=float, required=True, help="The clip overlap in seconds to use when clipping the audiofiles. Cannot be greater than or equal to the '-clip_duration_seconds' argument.")
    parser.add_argument("--seed_value", type=int, default=42, help="The value to pass to PytorchLightning.seed_everything() call")
    num_workers_default = int(os.environ.get("SLURM_CPUS_ON_NODE", default=1))
    parser.add_argument("--num_workers", type=int, default=num_workers_default, help="Number of dataloader workers to use")
    args = parser.parse_args()
    args.checkpoint = pathlib.Path(args.checkpoint).resolve()
    args.output_dir = pathlib.Path(args.output_dir).resolve()
    return args

if __name__ == "__main__":
    hyperparams = init()
    main(hyperparams)
