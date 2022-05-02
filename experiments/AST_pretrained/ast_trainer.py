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
        n_time_frames: int, 
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
        self.n_time_frames = n_time_frames
        self.device = device
        self.fstride = fstride
        self.tstride = tstride
        self.imagenet_pretrain = imagenet_pretrain
        self.audioset_pretrain = audioset_pretrain
        self.model_size = model_size
    
    def instantiate(self) -> torch.nn.Module:
        self.logger.log("Instantiating model...")
        model = ASTWrapper(
            logger_factory=self.logger_factory,
            activation_func = None, # Because loss function is BCEWithLogitsLoss which includes sigmoid activation.
            label_dim=self.n_model_outputs,
            fstride=self.fstride, 
            tstride=self.tstride, 
            input_fdim=self.n_mels, 
            input_tdim=self.n_time_frames, 
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
    batch_size,
    epochs,
    lossfunction,
    optimizer_type,
    optimizer_args,
    optimizer_kwargs,
    num_workers,
    kfolds,
    n_model_outputs,
    verbose,
    device,
    n_time_frames,
    n_mels,
    hop_length,
    n_fft,
    scale_melbands,
    classification_threshold,
    clip_length_samples,
    clip_overlap_samples,
    verification_dataset_limit,
    fstride,
    tstride,
    imagenet_pretrain,
    audioset_pretrain,
    model_size,
    tracker
):
    logger = logger_factory.create_logger()

    worker = Binworker(
        pool_ref=multiprocessing.pool.Pool,
        n_processes=num_workers,
        timeout_seconds=None
    )

    model_provider = AstModelProvider(logger_factory=logger_factory,
        n_model_outputs=n_model_outputs,
        n_mels=n_mels,
        n_time_frames=n_time_frames,
        device=device,
        fstride=fstride,
        tstride=tstride,
        imagenet_pretrain=imagenet_pretrain,
        audioset_pretrain=audioset_pretrain,
        model_size=model_size
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
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        scale_melbands=scale_melbands,
        verbose=verbose
    )

    verification_dataset_provider = VerificationDatasetProvider(
        clipped_dataset=clipped_dataset,
        limit=verification_dataset_limit,
        balancer=CachedDatasetBalancer(
            dataset=clipped_dataset,
            logger_factory=logger_factory,
            worker=worker,
            verbose=verbose
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
        verbose=verbose
    )

    verification_tracker = WandbTracker(
        logger_factory=logger_factory,
        name=f"AST verification",
        tags=["verification", "ast"],
        note="verification run for AST model, disregard any results from this run."
    )
    
    folder = BalancedKFolder(
        n_splits=kfolds,
        shuffle=True, 
        random_state=None,
        balancer_ref=DatasetBalancer,
        balancer_args=(),
        balancer_kwargs={"logger_factory": logger_factory, "worker": worker,"verbose": verbose}
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
        batch_size=batch_size,
        epochs=epochs,
        lossfunction=lossfunction,
        num_workers=num_workers
    )

    evaluator = Evaluator(
        logger_factory=logger_factory, 
        batch_size=batch_size,
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
        n_fold_workers=min(kfolds, multiprocessing.cpu_count())
    )
    cv.kfoldcv()
    logger.log("Verification run complete!")
    verification_tracker.run.finish()
    wandb.finish()

def proper(
    logger_factory,
    batch_size,
    epochs,
    lossfunction,
    optimizer_type,
    optimizer_args,
    optimizer_kwargs,
    num_workers,
    kfolds,
    n_model_outputs,
    verbose,
    device,
    n_time_frames,
    n_mels,
    hop_length,
    n_fft,
    scale_melbands,
    classification_threshold,
    clip_length_samples,
    clip_overlap_samples,
    proper_dataset_limit,
    fstride,
    tstride,
    imagenet_pretrain,
    audioset_pretrain,
    model_size,
    tracker
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
        n_mels=n_mels,
        n_time_frames=n_time_frames,
        device=device,
        fstride=fstride,
        tstride=tstride,
        imagenet_pretrain=imagenet_pretrain,
        audioset_pretrain=audioset_pretrain,
        model_size=model_size,
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
        verbose=verbose
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
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        scale_melbands=scale_melbands,
        verbose=verbose
    )

    complete_tensordataset = TensorAudioDataset(
        dataset=clipped_dataset,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor,
        logger_factory=logger_factory
    )

    complete_tensordataset.example_shape()

    complete_dataset_provider = BasicDatasetProvider(dataset=complete_tensordataset)

    folder = BalancedKFolder(
        n_splits=kfolds,
        shuffle=True, 
        random_state=None,
        balancer_ref=DatasetBalancer,
        balancer_args=(),
        balancer_kwargs={"logger_factory": logger_factory, "worker": worker,"verbose": verbose}
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
        batch_size=batch_size,
        epochs=epochs,
        lossfunction=lossfunction,
        num_workers=num_workers,
        device=device
    )

    dataset_verifier = BinaryTensorDatasetVerifier(
        logger_factory=logger_factory,
        worker=worker,
        verbose=verbose
    )

    evaluator = Evaluator(
        logger_factory=logger_factory, 
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )

    saver = Saver(logger_factory=logger_factory)

    logger.log(f"Beginning {kfolds}-fold cross evaluation...")
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
        n_fold_workers=min(kfolds, multiprocessing.cpu_count())
    )
    cv.kfoldcv()

def main():
    batch_size = 16
    epochs = 3
    lossfunction = torch.nn.BCEWithLogitsLoss()
    num_workers = min(multiprocessing.cpu_count(), 32) # DataLoader has max workers of 32
    lr = 0.00001
    weight_decay=5e-7 # Same as for AST paper code: https://github.com/YuanGongND/ast/blob/master/src/traintest
    betas=(0.95, 0.999) # Same as for AST paper code: https://github.com/YuanGongND/ast/blob/master/src/traintest

    optimizer_type=torch.optim.Adam
    optimizer_args=()
    optimizer_kwargs=dict(lr=lr, weight_decay=weight_decay, betas=betas)

    kfolds = 5
    n_model_outputs = 2
    verbose = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    n_mels = 1024
    hop_length = 256
    n_fft = 16384
    scale_melbands=False
    classification_threshold = 0.5

    fstride=10
    tstride=10
    imagenet_pretrain=True
    audioset_pretrain=False # by setting this to false, we can choose the input dimensions as we like
    model_size="base384" # [tiny224, small224, base224, base384]

    sr = 128000
    clip_length_samples = int(sr * 30.0) # ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, n_mels, n_time_frames)
    clip_overlap_samples = int(sr * 10.0) #int(clip_length_samples * 0.25)

    tracking_name=f"AST Pretrained Unfrozen"
    note="AST Pretrained Unfrozen"
    tags=["AST"]

    ### Only used for limited dataset during verification run ###
    verification_dataset_limit = 42
    proper_dataset_limit = 0.7 # percentage of clips in proper dataset

    logger_factory = LoggerFactory(
        logger_type=Logger, 
        logger_args=(LogFormatter(),)
    )

    logger = logger_factory.create_logger()
    if not torch.cuda.is_available() and "idun" in socket.gethostname():
        logger.log(f"Cuda is not available in the current environment (hostname: {repr(socket.gethostname())}), aborting...")
        exit(1)
    
    tracker = WandbTracker(
        logger_factory=logger_factory,
        name=tracking_name,
        note=note,
        tags=tags,
        batch_size=batch_size,
        epochs=epochs,
        lossfunction=lossfunction,
        num_workers=num_workers,
        kfolds=kfolds,
        verbose=verbose,
        device=device,
        optimizer_info=dict(
            type=str(optimizer_type.__name__),
            lr=lr,
            weight_decay=weight_decay,
            betas=str(betas)
        ),
        dataset_info=dict(
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            scale_melbands=scale_melbands,
            clip_length_samples=clip_length_samples,
            clip_overlap_samples=clip_overlap_samples,
            dataset_limit=proper_dataset_limit,
        ),
        model_info=dict(
            fstride=fstride,
            tstride=tstride,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            n_model_outputs=n_model_outputs,
        )
    )

    verify(
        logger_factory=logger_factory,
        batch_size=batch_size,
        epochs=epochs,
        lossfunction=lossfunction,
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_kwargs=optimizer_kwargs,
        num_workers=num_workers,
        kfolds=kfolds,
        n_model_outputs=n_model_outputs,
        verbose=verbose,
        device=device,
        n_time_frames=n_time_frames,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        scale_melbands=scale_melbands,
        classification_threshold=classification_threshold,
        clip_length_samples=clip_length_samples,
        clip_overlap_samples=clip_overlap_samples,
        verification_dataset_limit=verification_dataset_limit,
        fstride=fstride,
        tstride=tstride,
        imagenet_pretrain=imagenet_pretrain,
        audioset_pretrain=audioset_pretrain,
        model_size=model_size,
        tracker=tracker
    )
    

    proper(
        logger_factory=logger_factory,
        batch_size=batch_size,
        epochs=epochs,
        lossfunction=lossfunction,
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_kwargs=optimizer_kwargs,
        num_workers=num_workers,
        kfolds=kfolds,
        n_model_outputs=n_model_outputs,
        verbose=verbose,
        device=device,
        n_time_frames=n_time_frames,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        scale_melbands=scale_melbands,
        classification_threshold=classification_threshold,
        clip_length_samples=clip_length_samples,
        clip_overlap_samples=clip_overlap_samples,
        proper_dataset_limit=proper_dataset_limit,
        fstride=fstride,
        tstride=tstride,
        imagenet_pretrain=imagenet_pretrain,
        audioset_pretrain=audioset_pretrain,
        model_size=model_size,
        tracker=tracker,
    )

if __name__ == "__main__":
    main()