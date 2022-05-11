#!/usr/bin/env python3
import argparse
import os
import re
import sys
import pathlib
from typing import Mapping, Iterable, Optional
import multiprocessing

import git
import torch
import torch.utils.data
import torchmetrics
import wandb
from sklearn.model_selection import train_test_split
from test_tube.hpc import SlurmCluster, HyperOptArgumentParser, AbstractCluster
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from interfaces import ILoggerFactory, ITensorAudioDataset, IDatasetBalancer
from tracking.logger import BasicLogger
from tracking.loggerfactory import LoggerFactory

from models.AST.ASTWrapper import ASTWrapper

from datasets.tensordataset import TensorAudioDataset

from experiments.SupervisedAST.initdata import create_tensorset

from metrics import customwandbplots
from tracking.datasettracker import track_dataset

class AstLightningWrapper(pl.LightningModule):
    """
    AST (Audio Spectrogram Transformer) pretraining wrapper. Enables custom activation
    """
    def __init__(
        self, 
        logger_factory: ILoggerFactory,
        learning_rate: float,
        weight_decay: float,
        betas: Iterable[float],
        batch_size: int,
        activation_func: torch.nn.Module = None, 
        n_model_outputs=2, 
        fstride=10, 
        tstride=10, 
        input_fdim=128, 
        input_tdim=1024, 
        imagenet_pretrain=True, 
        audioset_pretrain=False, 
        model_size='base384',
        verbose=True) -> None:

        super().__init__()
        self._ast = ASTWrapper(
            logger_factory=logger_factory,
            activation_func=torch.nn.Sigmoid(),
            label_dim=n_model_outputs, 
            fstride=fstride, 
            tstride=tstride, 
            input_fdim=input_fdim, 
            input_tdim=input_tdim, 
            imagenet_pretrain=imagenet_pretrain, 
            audioset_pretrain=audioset_pretrain, 
            model_size=model_size, 
            verbose=verbose
        )
        self._activation = activation_func
        self._lossfunc = torch.nn.BCEWithLogitsLoss()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._betas = betas
        self._printlogger = logger_factory.create_logger()
        self._batch_size = batch_size
        self._accuracy = torchmetrics.Accuracy(num_classes=2)
        self._aucroc = torchmetrics.AUROC(num_classes=2)
        self._precision = torchmetrics.Precision(num_classes=2)
        self._recall = torchmetrics.Recall(num_classes=2)
        self._average_precision = torchmetrics.AveragePrecision(num_classes=2)
        self._f1 = torchmetrics.F1Score(num_classes=2)
        self._confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True, threshold=0.5) # Dont normalize here, send raw values to wandb and normalize in GUI
        self._verbose = verbose
        self.save_hyperparameters()

    def update_metrics(self, stepname, Yhat, Y):
        self._accuracy(Yhat.float(), Y.int())
        self._aucroc.update(Yhat.float(), Y.int())
        self._precision.update(Yhat.float(), Y.int())
        self._recall.update(Yhat.float(), Y.int())
        self._average_precision.update(Yhat.float(), Y.int())
        self._f1.update(Yhat.float(), Y.int())
        
        self.log(f"{stepname}_accuracy", self._accuracy, on_step=False, on_epoch=True)
        self.log(f"{stepname}_aucroc", self._aucroc, on_step=False, on_epoch=True)
        self.log(f"{stepname}_precision", self._precision, on_step=False, on_epoch=True)
        self.log(f"{stepname}_recall", self._recall, on_step=False, on_epoch=True)
        self.log(f"{stepname}_average_precision", self._average_precision, on_step=False, on_epoch=True)
        self.log(f"{stepname}_f1", self._f1, on_step=False, on_epoch=True)

        self._confusion_matrix.update(Yhat.float(), Y.int())

    def log_confusion_matrix(self, stepname):
        confusion = self._confusion_matrix.compute() # has shape (2, 2, 2)
        biophonic_confusion = confusion[0]
        anthropogenic_confusion = confusion[1]
        
        self.logger.experiment.log({f"{stepname}_bio_confusion_matrix": customwandbplots.confusion_matrix(biophonic_confusion, class_names=["not bio", "bio"], title=f"{stepname} confusion matrix (biophonic)")})
        self.logger.experiment.log({f"{stepname}_anth_confusion_matrix": customwandbplots.confusion_matrix(anthropogenic_confusion, class_names=["not anth", "anth"], title=f"{stepname} confusion matrix (anthropogenic)")})

    def forward(self, X):
        """Expect batch to have shape (batch_size, 1, n_mel_bands, n_time_frames)"""
        # AST.py expects input to have shape (batch_size, n_time_fames, n_mel_bans), swap third and fourth axis of X and squeeze second axis
        return self._ast(X)

    def training_step(self, batch, batch_idx):
        X, Y = batch # [batch_size, 1, n_mels, n_time_frames], [batch_size, 2]
        Yhat = self.forward(X) # [batch_size, 2]
        loss = self._lossfunc(Yhat, Y)
        self.log("train_loss", loss)
        return dict(loss=loss) # these are sent as input to training_epoch_end    

    def test_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self._lossfunc(Yhat, Y)
        self.update_metrics("test", Yhat, Y)
        return dict(loss=loss)

    def test_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix("test")

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Yhat = self.forward(X)
        loss = self._lossfunc(Yhat, Y)
        self.update_metrics("val", Yhat, Y)
        return dict(loss=loss)
    
    def validation_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix("val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self._learning_rate, betas=self._betas, weight_decay=self._weight_decay)
        return dict(optimizer=optimizer)

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ITensorAudioDataset, subset: Iterable[int], limit: int = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.limit = limit if limit is not None else len(subset)
        self.subset = subset[:self.limit]

    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.subset[index]]

class ClippedGliderDataModule(pl.LightningDataModule):
    def __init__(self, tensorset: TensorAudioDataset, balancer: IDatasetBalancer, batch_size: int, train_limit: int = None, val_limit: int = None, test_limit: int = None) -> None:
        super().__init__()
        self.tensorset = tensorset
        self.balancer = balancer
        self.batch_size = batch_size
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.val_limit = val_limit

    def setup(self, stage: Optional[str] = None) -> None:
        self.balancer.shuffle()

        self.eval_only_indeces = self.balancer.eval_only_indeces()
        self.train_indeces = self.balancer.train_indeces()

        distributions = self.balancer.label_distributions()
        n_per_label = {label: len(indeces) for label, indeces in distributions.items()}


        train_val_percentage = 0.8
        test_percentage = 1 - train_val_percentage
        
        n_for_training = int(len(self.train_indeces) * train_val_percentage)
        n_from_eval_only = int(len(self.eval_only_indeces) * test_percentage)

        # Indeces used for training and validation
        self.train_and_val_part = self.train_indeces[:n_for_training]
        self.train_indeces, self.val_indeces = train_test_split(self.train_and_val_part, test_size=0.2)

        # Indeces for testing
        test_part = self.train_indeces[n_for_training:] # These are balanced
        unbalanced_parts = self.eval_only_indeces[:n_from_eval_only] # These are unbalanced
        self.test_indeces = np.concatenate([test_part, unbalanced_parts]) # This way label distribution is maintained for testset

        # Train-, val- and testsets as subset datasets
        self.train = SubsetDataset(dataset=self.tensorset, subset=self.train_indeces, limit=self.train_limit)
        self.val = SubsetDataset(dataset=self.tensorset, subset=self.val_indeces, limit=self.val_limit)
        self.test = SubsetDataset(dataset=self.tensorset, subset=self.test_indeces, limit=self.test_limit)

        to_log = {
            "loader_sizes": {
                "train_loader_size": len(self.train),
                "val_loader_size": len(self.val),
                "test_loader_size": len(self.test)
            },
            "label_distributions": n_per_label,
            "tensorset_size": len(self.tensorset)
        }
        wandb.config.update(to_log)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

def main(hyperparams, *slurmargs):
    hyperparams.betas = [float(val) for val in hyperparams.betas.split(", ")]
    hyperparams.tracking_tags = [tag for tag in hyperparams.tracking_tags.split(", ")]
    sr = 128000
    fdim = hyperparams.nmels
    tdim = int((hyperparams.clip_duration_seconds * sr / hyperparams.hop_length) + 1)

    logger_factory = LoggerFactory(logger_type=BasicLogger)
    mylogger = logger_factory.create_logger()
    mylogger.log("Received hyperparams:", vars(hyperparams))
    
    logger = WandbLogger(
        save_dir=str(config.HOME_PROJECT_DIR.absolute()),
        offline=False,
        project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
        entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
        tags=hyperparams.tracking_tags,
        notes=hyperparams.tracking_note,
    )

    model = AstLightningWrapper(
        logger_factory=logger_factory,
        learning_rate=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
        betas=hyperparams.betas,
        batch_size=hyperparams.batch_size, # Only required for auto_scaling of batch_size
        activation_func=None,
        n_model_outputs=2,
        fstride=hyperparams.fstride,
        tstride=hyperparams.tstride,
        input_fdim=fdim,
        input_tdim=tdim,
        imagenet_pretrain=hyperparams.imagenet_pretrain,
        audioset_pretrain=hyperparams.audioset_pretrain,
        model_size=hyperparams.model_size,
        verbose=hyperparams.verbose,
    )

    tensorset, balancer = create_tensorset(
        logger_factory=logger_factory,
        nfft=hyperparams.nfft,
        nmels=hyperparams.nmels,
        hop_length=hyperparams.hop_length,
        clip_duration_seconds=hyperparams.clip_duration_seconds,
        clip_overlap_seconds=hyperparams.clip_overlap_seconds,
    )

    dataset = ClippedGliderDataModule(
        tensorset=tensorset, 
        balancer=balancer,
        batch_size=hyperparams.batch_size,
        train_limit=None,
        test_limit=None,
        val_limit=None,
    )

    prod = "idun" in os.uname().nodename
    trainer = pl.Trainer(
        accelerator="gpu" if prod else None, 
        devices=hyperparams.num_gpus if prod else None, 
        num_nodes=hyperparams.num_nodes if prod else None,
        # strategy="ddp",
        max_epochs=hyperparams.epochs,
        logger=logger,
        # auto_scale_batch_size=True # Not supported for DDP per. vXXX: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#batch-size-finder
    )
    
    logger.watch(model)
    wandb.config.update(vars(hyperparams))
    track_dataset(tensorset, n_examples=50)

    # trainer.tune(model, datamodule=dataset)
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)

def init():
    parser = HyperOptArgumentParser(strategy="random_search")
    parser.add_argument("--n_experiments", type=int, default=1)
    # Model params
    parser.opt_list("--learning_rate", type=float, tunable=True, options=[0.001, 0.0005, 0.00001])
    parser.opt_range("--weight_decay", type=float, tunable=True, low=1e-7, high=1e-2, nb_samples=8)
    parser.add_argument("--betas", type=str, default="0.95, 0.999") # Because test-tube wont handle args with nargs="+", split by "," and cast to float later
    parser.opt_list("--fstride", type=int, default=10, tunable=True, options=[10, 16])
    parser.opt_list("--tstride", type=int, default=10, tunable=True, options=[10, 16])
    parser.add_argument("--imagenet_pretrain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audioset_pretrain", action=argparse.BooleanOptionalAction, default=False)
    parser.opt_list("--model_size", type=str, tunable=True, options=["tiny224", "small224", "base224", "base384"])
    # Data params
    parser.opt_list("--batch_size", type=int, tunable=True, options=[4, 8, 16, 32, 64])
    
    parser.opt_list("--nmels", type=int, tunable=True, default=128, options=[128, 64, 256, 512, 1024])
    parser.opt_list("--nfft", type=int, tunable=True, default=3200, options=[3200, 4096, 8192])
    parser.opt_list("--hop_length", type=int, tunable=True, default=1280, options=[1280, 512, 1024, 2048, 4096])
    parser.add_argument("--clip_duration_seconds", type=float, default=10.0)
    parser.add_argument("--clip_overlap_seconds", type=float, default=4.0) 

    # Training params
    parser.opt_list("--epochs", type=int, tunable=True, options=[3, 10, 20, 50, 100])
    
    # Tracking params
    parser.add_argument("--tracking_note", type=str, required=False)
    parser.add_argument("--tracking_tags", type=str, required=False, default="AST")
    parser.add_argument("--track_n_examples", type=int, default=50)
    # Other params
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--num_gpus", type=int, default=2) # Number of GPUs per run
    parser.add_argument("--num_nodes", type=int, default=1) # Number of compute nodes per run
    parser.add_argument("--num_cpus", type=int, default=32) # Number of CPUs per run
    parser.add_argument("--mem", type=int, default=42000) # Allocated memory in MB for each run

    args = parser.parse_args()
    
    env = "Prod" if "idun" in os.uname().nodename else "Dev"
    args.tracking_tags += f", {env}"
    if args.model_size is not None: args.tracking_tags += f", {args.model_size}"
    
    args.tracking_tags += f", {'No-' if not args.audioset_pretrain else ''}AudioSet"
    args.tracking_tags += f", {'No-' if not args.imagenet_pretrain else ''}ImageNet"
    return args

def override_slurm_cmd_command(cluster: SlurmCluster) -> SlurmCluster:
    """According to the HPC department at NTNU, responsible for the Idun cluster, sbatch scripts should not contain any 'srun' calls.
    The default SlurmCluster implementation however does call srun. Therefore we must remove any 'srun' strings from the SlurmCluster __build_slurm_command method
    If in contact with the NTNU Hjelp in the future, reference casenumber NTNU0505485

    Args:
        cluster (SlurmCluster): The original SlurmCluster object instance

    Returns:
        SlurmCluster: The same SlurmCluster instance, but with it's __build_slurm_command method overridden. 
    """
    def overridden(*args, **kwargs): 
        cmd = helper(*args, **kwargs)
        new = re.sub(r"srun\s*", "", cmd)
        return new
    
    helper = cluster._SlurmCluster__build_slurm_command
    cluster._SlurmCluster__build_slurm_command = overridden
    return cluster

def start_slurmjobs(hyperparams):
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=config.SLURM_LOGS_DIR,
        python_cmd="python"
    )
    cluster = override_slurm_cmd_command(cluster)
    
    cluster.notify_job_status(email=os.environ.get("SLURM_NOTIFY_EMAIL_ADDRESS"), on_done=True, on_fail=True)
    cluster.job_time = "00-48:00:00"
    cluster.minutes_to_checkpoint_before_walltime = 2
    cluster.per_experiment_nb_cpus = hyperparams.num_cpus # Number of CPUs each experiment/run gets
    cluster.per_experiment_nb_gpus = hyperparams.num_gpus # Number of GPUs each experiment/run gets
    cluster.per_experiment_nb_nodes = hyperparams.num_nodes # Number of Compute Nodes each experminet/run gets
    cluster.memory_mb_per_node = hyperparams.mem
    # cluster.gpu_type = '1080ti'

    cluster.add_command("module purge")
    cluster.add_command("module load Anaconda3/2020.07")
    cluster.add_command("module load PyTorch/1.8.1-fosscuda-2020b")
    cluster.add_command("module load NCCL/2.8.3-CUDA-11.1.1")

    cluster.add_command("conda init --all")
    cluster.add_command("source ~/.bashrc")
    cluster.add_command("conda activate TDT4900")
    cluster.add_command("conda info --envs")
    cluster.add_command("export NCCL_DEBUG=INFO")
    cluster.add_command("export PYTHONFAULTHANDLER=1")

    initdata_script = pathlib.Path(__file__).parent.joinpath("initdata.py").absolute()
    cluster.add_command(f"python {initdata_script} -nmels {hyperparams.nmels} -hop_length {hyperparams.hop_length} -nfft {hyperparams.nfft} -clip_duration_seconds {hyperparams.clip_duration_seconds} -clip_overlap_seconds {hyperparams.clip_overlap_seconds}")
    
    cluster.add_slurm_cmd(cmd="account", value="ie-idi", comment="Accounting job to use for the job")
    cluster.add_slurm_cmd(cmd="partition", value="GPUQ", comment="Job partition (CPUQ/GPUQ)")

    cluster.optimize_parallel_cluster_gpu(main, nb_trials=hyperparams.n_experiments, job_name="AST")

if __name__ == "__main__":
    hyperparams = init()
    start_slurmjobs(hyperparams)