#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib

import git
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from experiments.SupervisedAST.model import AstLightningWrapper, ModelSize
from datasets.datamodule import ClippedGliderDataModule
from tracking.datasettracker import track_dataset

def main(hyperparams):
    # Tracking issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/11380
    pl.seed_everything(hyperparams.seed_value)
    sr = 128000
    fdim = hyperparams.nmels
    tdim = int((hyperparams.clip_duration_seconds * sr / hyperparams.hop_length) + 1)
    print("Received hyperparams:", vars(hyperparams))

    logger = WandbLogger(
        save_dir=str(config.HOME_PROJECT_DIR.absolute()),
        offline=False,
        project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
        entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
        config=vars(hyperparams), # These are added to wandb.init call as part of the config,
        tags=hyperparams.tracking_tags,
        notes=hyperparams.tracking_notes,
        name=hyperparams.tracking_name
    )

    dataset = ClippedGliderDataModule(
        batch_size=hyperparams.batch_size,
        nfft=hyperparams.nfft,
        nmels=hyperparams.nmels,
        hop_length=hyperparams.hop_length,
        clip_duration_seconds=hyperparams.clip_duration_seconds,
        clip_overlap_seconds=hyperparams.clip_overlap_seconds,
        num_workers=hyperparams.num_workers
    )
    track_dataset(logger, dataset, n_examples=hyperparams.track_n_examples)

    model = AstLightningWrapper(
        learning_rate=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
        betas=hyperparams.betas,
        batch_size=hyperparams.batch_size, # Only required for auto_scaling of batch_size
        n_model_outputs=2,
        fstride=hyperparams.fstride,
        tstride=hyperparams.tstride,
        input_fdim=fdim,
        input_tdim=tdim,
        imagenet_pretrain=hyperparams.imagenet_pretrain,
        audioset_pretrain=hyperparams.audioset_pretrain,
        model_size=hyperparams.model_size,
        class_names=dataset.class_names(),
        verbose=hyperparams.verbose
    )

    checkpoints_dir = config.LIGHTNING_CHECKPOINT_PATH.joinpath("ast").absolute() # Must separate ast and ssast checkpoints, because they are stored with the same filenames by pytorch_lightning...
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=False, exist_ok=False)

    trainer = pl.Trainer(
        accelerator=hyperparams.accelerator, 
        devices=hyperparams.num_gpus,
        num_nodes=hyperparams.num_nodes,
        strategy=hyperparams.strategy,
        max_epochs=hyperparams.epochs,
        logger=logger,
        weights_save_path=str(checkpoints_dir),
        fast_dev_run=hyperparams.dev_run,
        overfit_batches=hyperparams.overfit_batches,
        limit_train_batches=hyperparams.limit_train_batches,
        limit_test_batches=hyperparams.limit_test_batches,
        limit_val_batches=hyperparams.limit_val_batches,
        default_root_dir=str(config.LIGHTNING_CHECKPOINT_PATH.absolute()),
        log_every_n_steps=hyperparams.log_every_n_steps
    )
    logger.watch(model)    
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)

def float_or_int_argtype(value):
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        raise argparse.ArgumentTypeError(f"Argument '{value}' has incorrect type, expected 'float' or 'int' but received '{type(value)}' with value {value}")

def init():
    parser = argparse.ArgumentParser(
        description="Training script to perform supervised training of Audio Spectrogram Transformer (AST) on the GLIDER dataset."
    )
    # Model params
    parser.add_argument("-weight_decay", type=float, required=True, help="The weight decay to pass to the PyTorch optimizer")
    parser.add_argument("-learning_rate", type=float, required=True, help="The learning rate to pass to the PyTorch optimizer")
    parser.add_argument("-betas", type=float, nargs="+", required=True, help="The betas to pass to the PyTorch optimizer")
    parser.add_argument("-fstride", type=int, default=10, help="The frequency-dimension stride of the AST patches in 'mel-bands'. Patches have size (16x16), so an fstride of 10 results in patch overlap of 6 frames along the frequency dimension of the input spectrograms. Conversly an fstride of 16 results in no overlap between mel-bands.")
    parser.add_argument("-tstride", type=int, default=10, help="The time-dimension stride of the AST patches in 'time-frames'. Patches have size (16x16), so a tstride of 10 results in patch overlap of 6 frames along the time dimension of the input spectrograms. Conversly a tstride of 16 results in no overlap between time-frames.")
    parser.add_argument("--audioset_pretrain", action=argparse.BooleanOptionalAction, help=f"Wheter to instantiate the AST using weights pretrained on AudioSet (which themselves are pretrained on ImageNet). Will cause an exception if '--no-imagenet_pretrain' argument is also provided, as no model pretrained solely on audioset is currently supported. If set, the '--model_size' must be '{ModelSize.base384.value}'")
    parser.add_argument("--imagenet_pretrain", action=argparse.BooleanOptionalAction, help="Wheter to instantiate the AST using weights pretrained on ImageNet")
    
    parser.add_argument("-model_size", type=str, default=ModelSize.base384.value, choices=[ModelSize.tiny224.value, ModelSize.small224.value, ModelSize.base224.value, ModelSize.base384.value], help=f"Which pre-trained parameters/model size to instantiate the AST from. If '--no-audioset_pretrain' is also provided the only valid option is '{ModelSize.base384.value}'")

    # Data params
    parser.add_argument("-batch_size", type=int, required=True, help="The batch size to use during training, testing and evaluation")
    parser.add_argument("-nmels", type=int, required=True, help="The number of Mel-bands/filters to use when computing the Mel-spectrogram from the raw audio data. This argument determines the 'vertical' dimensions of the spectrograms.")
    parser.add_argument("-nfft", type=int, required=True, help="The size of window in number of samples to compute the FFT over during the Short-Time Fourier Transform algorithm. A larger window yields better frequency determination, but degrades time determination, and vice-versa.")
    parser.add_argument("-hop_length", type=int, required=True, help="The hop length in number of samples skip between successive windows when computing the Short-Time Fourier Transform of the raw audio. This, together with the '-nfft' argument and the number of samples (determined by sampling rate and '-clip_duration_seconds' argument) in the raw audio data, determines the size of the time-dimension of the resulting spectrograms. If '-hop_length' is equal to '-nfft' successive STFT windows will not overlap, if '-hop_length' < '-nfft' successive STFT windows will overlap by ('-nfft' - '-hop_length' samples)")
    parser.add_argument("-clip_duration_seconds", type=float, required=True, help="The clip duration in seconds to use when clipping the raw audiofiles. Clipping is done 'virtually' before actually reading any samples from the audiofile into memory, by using the known sampling rate and duration of each file. This is done to improve performance and memory efficiency, and such that we can compute the dataset size without first reading all the files, clipping them and aggregating the result.")
    parser.add_argument("-clip_overlap_seconds", type=float, required=True, help="The clip overlap in seconds to use when clipping the audiofiles. Cannot be greater than or equal to the '-clip_duration_seconds' argument.")

    # Training params
    parser.add_argument("-epochs", type=int, required=True, help="The maximum number of epochs to train for before performing testing. For every epoch the model is trained and validated.")

    default_strategy = "ddp" if os.environ.get("SLURM_JOBID", default=None) is not None else None
    parser.add_argument("--strategy", type=str, default=default_strategy, help="The strategy name passed to the PytorchLightning.Trainer instantiation")
    parser.add_argument("--accelerator", type=str, default="gpu", help="The accelerator name passed to the PytorchLightning.Trainer instantiation")

    # Tracking params
    parser.add_argument("--tracking_name", type=str, required=False, help="The WandB run name to use during the run. If not provided a WandB will generate a name automatically.")
    tracking_notes = os.environ.get("SLURM_JOB_NAME", default=None)
    parser.add_argument("--tracking_notes", type=str, default=tracking_notes, help="Any notes to use for the WandB run. Will by default use the 'SLURM_JOB_NAME' environment variable if available")
    parser.add_argument("--tracking_tags", type=str, nargs="+", required=False, help="Any WandB tags to use during the run in addition to the default tags ['AST', 'Prod' (if script is run in SLURM managed environment), model size value, 'AudioSet'/'No-AudioSet', 'ImageNet'/'No-ImageNet']")
    parser.add_argument("--track_n_examples", type=int, default=50)
    # Other params
    parser.add_argument("--verbose", action="store_true", default=False)

    num_gpus_default = int(os.environ.get("SLURM_GPUS_ON_NODE", default=-1))
    if num_gpus_default == -1:
        num_gpus_default = torch.cuda.device_count()
        
    parser.add_argument("--num_gpus", type=int, default=num_gpus_default, help="The number of GPUs to use during training. Defaults to the environment variable 'SLURM_GPUS_ON_NODE' if present, if not set and the environment variable is not found defaults to 'torch.cuda.device_count()'.")

    num_nodes_default = int(os.environ.get("SLURM_NNODES", default=1))
    parser.add_argument("--num_nodes", type=int, default=num_nodes_default, help="The number of compute nodes to use during training. Defaults to the environment vairable 'SLURM_NNODES' if present. If not set and the environment variable is not found and the '--strategy' argument is 'ddp' or 'ddp2' will raise an exception, if other strategy is used will default to 'None'")
    
    num_workers_default = int(os.environ.get("SLURM_CPUS_ON_NODE", default=0))
    parser.add_argument("--num_workers", type=int, default=num_workers_default, help="The number of workers per node to use. Defaults to value of the 'SLURM_CPUS_ON_NODE' environment variable if available or 0 if not available. If 0 the data will be loaded from the main process (for each GPU)")

    parser.add_argument("--dev_run", action="store_true", default=False, help="If this flag is provided the PytorchLightning.Trainer instantiation will receive 'fast_dev_run=True'")
    parser.add_argument("--limit_train_batches", type=int, default=1.0, required=False, help="The number of instances of the training dataset to use. If not provided will use the entire training dataset.")
    parser.add_argument("--limit_test_batches", type=int, default=1.0, required=False, help="The number of instances of the testing dataset to use. If not provided will use the entire testing dataset.")
    parser.add_argument("--limit_val_batches", type=int, default=1.0, required=False, help="The number of instances of the validation dataset to use. If not provided will use the entire validation dataset.")
    parser.add_argument("--overfit_batches", type=float_or_int_argtype, default=0.0, required=False, help="The PytorchLightning.Trainer(overfit_batches) argument value. If and integer is provided will use that number of batches to overfit on, if a float value is provided will use that fraction of the training set to overfit on. Usefull for debugging. Defaults to 0.0")
    parser.add_argument("--seed_value", type=int, default=42, help="The value to pass to PytorchLightning.seed_everything() call")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="The log interval in number of training steps. Will be passed to Trainer instantiation as PytorchLightning.Trainer(log_every_n_steps=args.log_every_n_steps)")

    args = parser.parse_args()
    args.betas = tuple(args.betas)
    default_tags = [
        'AST',
        'Prod' if len([key for key, value in os.environ.items() if "SLURM" in key]) > 0 and "idun" in os.uname().nodename else 'Dev', # if running on cluster, and in SLURM managed environment, the environment tag should be Prod
        'ImageNet' if args.imagenet_pretrain else 'No-ImageNet',
        'Audioset' if args.audioset_pretrain else 'No-Audioset',
        args.model_size
    ]
    if args.tracking_tags is None:
        args.tracking_tags = default_tags
    else:
        args.tracking_tags += default_tags

    if args.num_nodes is None and args.strategy is not None and args.strategy.lower().strip() in ["ddp", "ddp2"]:
        raise Exception(f"No 'num_nodes' argument was not provided, and no usable value could be inferred. Strategy was {args.strategy} and environment variable 'SLURM_NNODES' was not found.")

    return args

if __name__ == "__main__":
    hyperparams = init()
    main(hyperparams)
