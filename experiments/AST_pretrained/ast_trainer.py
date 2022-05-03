#!/usr/bin/env python3
import multiprocessing
import multiprocessing.pool
import sys
import pathlib
from typing import Mapping
import socket

import git
import torch
import wandb

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ILoggerFactory, IModelProvider

from tracking.logger import Logger, LogFormatter
from tracking.saver import Saver
from tracking.loggerfactory import LoggerFactory
from tracking.wandb_tracker import WandbTracker

from models.trainer import Trainer
from models.evaluator import Evaluator
from models.optimizer import AdamaxProvider, GeneralOptimizerProvider
from models.modelprovider import DefaultModelProvider
from models.AST.ASTWrapper import ASTWrapper

from datasets.folder import BasicKFolder
from datasets.balancing import BalancedKFolder, DatasetBalancer, CachedDatasetBalancer
from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import BalancedKFolder
from datasets.limiting import DatasetLimiter, ProportionalDatasetLimiter
from datasets.tensordataset import TensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from datasets.provider import BasicDatasetProvider, VerificationDatasetProvider
from datasets.verifier import BinaryTensorDatasetVerifier
from datasets.binjob import Binworker
from models.crossevaluator import CrossEvaluator

from metrics import BinaryMetricComputer

import config

class AstModelProvider(IModelProvider):
    def __init__(
        self,
        logger_factory: ILoggerFactory,
        n_model_outputs: int, 
        n_mels: int, 
        n_fft: int,
        hop_length: int,
        sr: int,
        clip_length_samples: int,
        clip_overlap_samples: int,
        device: str,
        fstride: int,
        tstride: int,
        imagenet_pretrain: bool,
        audioset_pretrain: bool,
        model_size: str) -> None:
        
        super().__init__()
        self.logger_factory = logger_factory
        self.logger = logger_factory.create_logger()
        self.n_model_outputs = n_model_outputs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.clip_length_samples = clip_length_samples
        self.clip_overlap_samples = clip_overlap_samples
        self.device = device
        self.fstride = fstride
        self.tstride = tstride
        self.imagenet_pretrain = imagenet_pretrain
        self.audioset_pretrain = audioset_pretrain
        self.model_size = model_size
    
    def instantiate(self) -> torch.nn.Module:
        self.logger.log("Instantiating model...")

        t_dim = int((self.clip_length_samples / self.hop_length) + 1)
        # Alternatively:
        # frame_rate = self.sr / self.hop_length
        # clip_duration_seconds = self.sr * self.clip_length_samples
        # t_dim = int(frame_rate * clip_duration_seconds)
        f_dim = self.n_mels

        model = ASTWrapper(
            logger_factory=self.logger_factory,
            activation_func = None, # Because loss function is BCEWithLogitsLoss which includes sigmoid activation.
            label_dim=self.n_model_outputs,
            fstride=self.fstride, 
            tstride=self.tstride, 
            input_fdim=t_dim, 
            input_tdim=f_dim, 
            imagenet_pretrain=self.imagenet_pretrain, 
            audioset_pretrain=self.audioset_pretrain, 
            model_size=self.model_size,
            verbose=True
        )
        # self.logger.log("Freezing pre-trained model parameters...")
        # model.freeze_pretrained_parameters()
        # self.logger.log("Parameter freeze complete!")

        if torch.cuda.device_count() > 1:
            # Use all the available GPUs with DataParallel
            model = torch.nn.DataParallel(model)
        
        model.to(self.device)
        self.logger.log("Model instantiated!")
        return model

    @property
    def properties(self) -> Mapping[str, any]:
        props = {
            "n_model_outputs": self.n_model_outputs,
            "n_mels": self.n_mels,
            "n_time_frames": self.n_time_frames,
            "device": self.device,
        }
        return props

def verify(
    logger_factory,
    lossfunction,
    optimizer_type,
    optimizer_args,
    optimizer_kwargs,
    num_workers,
    n_model_outputs,
    scale_melbands,
    classification_threshold,
    device,
    clip_length_samples,
    clip_overlap_samples,
    args
):
    logger = logger_factory.create_logger()

    worker = Binworker(
        pool_ref=multiprocessing.pool.Pool,
        n_processes=num_workers,
        timeout_seconds=None
    )

    model_provider = AstModelProvider(logger_factory=logger_factory,
        n_model_outputs=n_model_outputs,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=device,
        fstride=args.fstride,
        tstride=args.tstride,
        imagenet_pretrain=args.imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain,
        model_size=args.model_size
    )

    clipped_dataset = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=worker,
        clip_nsamples=clip_length_samples,
        overlap_nsamples=clip_overlap_samples,
    )

    label_accessor = BinaryLabelAccessor()
    feature_accessor = MelSpectrogramFeatureAccessor(
        logger_factory=logger_factory, 
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        scale_melbands=scale_melbands,
        verbose=args.verbose
    )

    verification_dataset_provider = VerificationDatasetProvider(
        clipped_dataset=clipped_dataset,
        limit=args.verification_dataset_limit,
        balancer=CachedDatasetBalancer(
            dataset=clipped_dataset,
            logger_factory=logger_factory,
            worker=worker,
            verbose=args.verbose
        ),
        randomize=True,
        balanced=True,
        feature_accessor=feature_accessor,
        label_accessor=label_accessor,
        logger_factory=logger_factory
    )

    dataset_verifier = BinaryTensorDatasetVerifier(
        logger_factory=logger_factory,
        worker=worker,
        verbose=args.verbose
    )

    verification_tracker = WandbTracker(
        logger_factory=logger_factory,
        name=f"AST verification",
        tags=["verification", "ast"],
        note="verification run for AST model, disregard any results from this run.",
        n_examples=3
    )
    
    folder = BalancedKFolder(
        n_splits=args.kfolds,
        shuffle=True, 
        random_state=None,
        balancer_ref=DatasetBalancer,
        balancer_args=(),
        balancer_kwargs={"logger_factory": logger_factory, "worker": worker,"verbose": args.verbose}
    )

    optimizer_provider = GeneralOptimizerProvider(
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_kwargs=optimizer_kwargs
    )

    metric_computer = BinaryMetricComputer(
        logger_factory=logger_factory,
        class_dict=clipped_dataset.classes(),
        threshold=classification_threshold
    )

    verification_trainer = Trainer(
        logger_factory=logger_factory,
        optimizer_provider=optimizer_provider,
        metric_computer=metric_computer,
        tracker=verification_tracker,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lossfunction=lossfunction,
        num_workers=num_workers
    )

    evaluator = Evaluator(
        logger_factory=logger_factory, 
        batch_size=args.batch_size,
        num_workers=num_workers,
        device=device
    )

    saver = Saver(logger_factory=logger_factory)

    #### Perform "verificaiton" run to check that everything is working as expected.
    logger.log("\n\n\n")
    logger.log("-------------------------------------------------------")
    logger.log("\n\n\n")
    logger.log("Performing verification run...")
    logger.log("\n\n\n")
    logger.log("-------------------------------------------------------")
    logger.log("\n\n\n")

    cv = CrossEvaluator(
        model_provider=model_provider,
        logger_factory=logger_factory,
        dataset_provider=verification_dataset_provider,
        dataset_verifier=dataset_verifier,
        tracker=verification_tracker,
        folder=folder,
        trainer=verification_trainer,
        evaluator=evaluator,
        metric_computer=metric_computer,
        saver=saver,
        n_fold_workers=1 #min(kfolds, multiprocessing.cpu_count())
    )
    cv.kfoldcv()
    logger.log("Verification run complete!")
    verification_tracker.run.finish()
    wandb.finish()

def proper(
    logger_factory,
    lossfunction,
    optimizer_type,
    optimizer_args,
    optimizer_kwargs,
    num_workers,
    n_model_outputs,
    scale_melbands,
    classification_threshold,
    clip_length_samples,
    clip_overlap_samples,
    tracker,
    args
    ):

    logger = logger_factory.create_logger()
    
    logger.log("\n\n\n")
    logger.log("-------------------------------------------------------")
    logger.log("\n\n\n")
    logger.log("Beginning proper cross evaluation!")
    logger.log("\n\n\n")
    logger.log("-------------------------------------------------------")
    logger.log("\n\n\n")

    worker = Binworker(
        pool_ref=multiprocessing.pool.Pool,
        n_processes=num_workers,
        timeout_seconds=None
    )

    model_provider = AstModelProvider(
        logger_factory=logger_factory,
        n_model_outputs=n_model_outputs,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device=args.device,
        fstride=args.fstride,
        tstride=args.tstride,
        imagenet_pretrain=args.imagenet_pretrain,
        audioset_pretrain=args.audioset_pretrain,
        model_size=args.model_size,
    )

    clipped_dataset = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=worker,
        clip_nsamples=clip_length_samples,
        overlap_nsamples=clip_overlap_samples,
    )

    balancer=CachedDatasetBalancer(
        dataset=clipped_dataset,
        logger_factory=logger_factory,
        worker=worker, 
        verbose=args.verbose
    )

    # limited_dataset = ProportionalDatasetLimiter(
    #     clipped_dataset,
    #     balancer=balancer,
    #     logger_factory=logger_factory,
    #     size=proper_dataset_limit
    # )

    label_accessor = BinaryLabelAccessor()
    feature_accessor = MelSpectrogramFeatureAccessor(
        logger_factory=logger_factory, 
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        scale_melbands=scale_melbands,
        verbose=args.verbose
    )

    complete_tensordataset = TensorAudioDataset(
        dataset=clipped_dataset,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor,
        logger_factory=logger_factory
    )

    complete_dataset_provider = BasicDatasetProvider(dataset=complete_tensordataset)

    folder = BalancedKFolder(
        n_splits=args.kfolds,
        shuffle=True, 
        random_state=None,
        balancer_ref=DatasetBalancer,
        balancer_args=(),
        balancer_kwargs={"logger_factory": logger_factory, "worker": worker,"verbose": args.verbose}
    )

    optimizer_provider = GeneralOptimizerProvider(
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_kwargs=optimizer_kwargs,
    )

    metric_computer = BinaryMetricComputer(
        logger_factory=logger_factory,
        class_dict=clipped_dataset.classes(),
        threshold=classification_threshold
    )

    trainer = Trainer(
        logger_factory=logger_factory,
        optimizer_provider=optimizer_provider,
        tracker=tracker,
        metric_computer=metric_computer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lossfunction=lossfunction,
        num_workers=num_workers,
        device=args.device
    )

    dataset_verifier = BinaryTensorDatasetVerifier(
        logger_factory=logger_factory,
        worker=worker,
        verbose=args.verbose
    )

    evaluator = Evaluator(
        logger_factory=logger_factory, 
        batch_size=args.batch_size,
        num_workers=num_workers,
        device=args.device
    )

    saver = Saver(logger_factory=logger_factory)

    logger.log(f"Beginning {args.kfolds}-fold cross evaluation...")
    label_distributions = {label: len(values) for label, values in balancer.label_distributions().items()}
    tracker.run.config.update(
        {
            "tensor_dataset_size": len(complete_tensordataset), 
            "input_shape": str(complete_tensordataset.example_shape()), 
            "output_shape": str(complete_tensordataset.label_shape()), 
            "label_distributions": label_distributions
        }
    )
    
    #### Perform the proper training run
    cv = CrossEvaluator(
        model_provider=model_provider,
        logger_factory=logger_factory,
        dataset_provider=complete_dataset_provider,
        dataset_verifier=dataset_verifier,
        tracker=tracker,
        folder=folder,
        trainer=trainer,
        evaluator=evaluator,
        metric_computer=metric_computer,
        saver=saver,
        n_fold_workers=1 #min(kfolds, multiprocessing.cpu_count())
    )
    cv.kfoldcv()

def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-epochs", type=int, required=True)
    parser.add_argument("-learning_rate", type=float, required=True)
    parser.add_argument("-weight_decay", type=float, required=True)
    parser.add_argument("-betas", type=float, nargs="+", required=True)
    parser.add_argument("-kfolds", type=int, required=True)
    parser.add_argument("-n_mels", type=int, required=True)
    parser.add_argument("-n_fft", type=int, required=True)
    parser.add_argument("-hop_length", type=int, required=True)
    parser.add_argument("-fstride", type=int, default=10)
    parser.add_argument("-tstride", type=int, default=10)
    parser.add_argument("-imagenet_pretrain", type=bool, required=True)
    parser.add_argument("-audioset_pretrain", type=bool, required=True)
    parser.add_argument("-model_size", type=str, choices=["tiny224", "small224", "base224", "base384"])
    parser.add_argument("-clip_duration_seconds", type=float, required=True)
    parser.add_argument("-clip_overlap_seconds", type=float, required=True)
    parser.add_argument("-tracking_name", type=str, required=True)
    parser.add_argument("-tracking_note", type=str, required=True)
    parser.add_argument("-tracking_tags", type=str, nargs="+", required=True)
    parser.add_argument("-track_n_examples", type=int, default=50)
    parser.add_argument("-verification_dataset_limit", type=int, default=42)
    parser.add_argument("-proper_dataset_limit", default=0.7)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()

def main(args):
    logger_factory = LoggerFactory(
        logger_type=Logger, 
        logger_args=(LogFormatter(),)
    )

    logger = logger_factory.create_logger()
    if not torch.cuda.is_available() and "idun" in socket.gethostname():
        logger.log(f"Cuda is not available in the current environment (hostname: {repr(socket.gethostname())}), aborting...")
        exit(1)
    logger.log(f"Running {pathlib.Path(__file__).name} with arguments:", vars(args))

    lossfunction = torch.nn.BCEWithLogitsLoss()
    num_workers = min(multiprocessing.cpu_count(), 32) # torch DataLoader has max workers of 32

    optimizer_type=torch.optim.Adam
    optimizer_args=()
    optimizer_kwargs=dict(lr=args.learning_rate, weight_decay=args.weight_decay, betas=tuple(args.betas))

    n_model_outputs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale_melbands=False
    classification_threshold = 0.5

    sr = 128000
    clip_length_samples = int(sr * args.clip_duration_seconds) # ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, n_mels, n_time_frames)
    clip_overlap_samples = int(sr * args.clip_overlap_seconds) #int(clip_length_samples * 0.25)
    
    tracker = WandbTracker(
        logger_factory=logger_factory,
        name=args.tracking_name,
        note=args.tracking_note,
        tags=args.tracking_tags,
        n_examples=args.track_n_examples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lossfunction=lossfunction,
        num_workers=num_workers,
        kfolds=args.kfolds,
        verbose=args.verbose,
        device=device,
        optimizer_info=dict(
            type=str(optimizer_type.__name__),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=str(args.betas)
        ),
        dataset_info=dict(
            n_mels=args.n_mels,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            scale_melbands=scale_melbands,
            clip_length_samples=clip_length_samples,
            clip_overlap_samples=clip_overlap_samples,
            dataset_limit=args.proper_dataset_limit,
        ),
        model_info=dict(
            fstride=args.fstride,
            tstride=args.tstride,
            imagenet_pretrain=args.imagenet_pretrain,
            audioset_pretrain=args.audioset_pretrain,
            model_size=args.model_size,
            n_model_outputs=n_model_outputs,
        )
    )

    verify(
        logger_factory=logger_factory,
        lossfunction=lossfunction,
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_kwargs=optimizer_kwargs,
        num_workers=num_workers,
        n_model_outputs=n_model_outputs,
        scale_melbands=scale_melbands,
        classification_threshold=classification_threshold,
        device=device,
        clip_length_samples=clip_length_samples,
        clip_overlap_samples=clip_overlap_samples,
        args=args
    )

    proper(
        logger_factory=logger_factory,
        lossfunction=lossfunction,
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_kwargs=optimizer_kwargs,
        num_workers=num_workers,
        n_model_outputs=n_model_outputs,
        scale_melbands=scale_melbands,
        classification_threshold=classification_threshold,
        clip_length_samples=clip_length_samples,
        clip_overlap_samples=clip_overlap_samples,
        tracker=tracker,
        args=args
    )

if __name__ == "__main__":
    args = init()
    main(args)