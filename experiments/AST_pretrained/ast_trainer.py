#!/usr/bin/env python3
from curses.ascii import CR
import multiprocessing
import sys
import pathlib
from tabnanny import verbose

import git
from sklearn import feature_extraction
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IModelProvider, ILoggerFactory, IDatasetVerifier, ICrossEvaluator, ITracker, IDatasetProvider, ITrainer, IEvaluator, IMetricComputer, IFolder, ISaver

from tracking.logger import Logger, LogFormatter
from tracking.tracker import Tracker
from tracking.saver import Saver
from tracking.loggerfactory import LoggerFactory
from tracking.sheets import SheetClient

from models.trainer import Trainer
from models.evaluator import Evaluator
from models.optimizer import AdamaxProvider, GeneralOptimizerProvider
from models.modelprovider import DefaultModelProvider
from models.AST.ASTWrapper import ASTWrapper

from datasets.folder import BasicKFolder
from datasets.balancing import BalancedKFolder, DatasetBalancer
from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import BalancedKFolder
from datasets.limiting import DatasetLimiter
from datasets.tensordataset import TensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor
from datasets.provider import BasicDatasetProvider, VerificationDatasetProvider
from datasets.verifier import BinaryTensorDatasetVerifier
from datasets.binjob import Binworker
from models.crossevaluator import CrossEvaluator

from metrics import BinaryMetricComputer

import config

def instantiate_model(
    n_model_outputs: int, 
    n_mels: int, 
    n_time_frames: int, 
    device: str):

    model = ASTWrapper(
        activation_func = None, # Because loss function is BCEWithLogitsLoss which includes sigmoid activation.
        label_dim=n_model_outputs,
        fstride=10, 
        tstride=10, 
        input_fdim=n_mels, 
        input_tdim=n_time_frames, 
        imagenet_pretrain=True, 
        audioset_pretrain=True, 
        model_size='base384',
        verbose=True
    )

    model.freeze_pretrained_parameters()

    if torch.cuda.device_count() > 1:
        # Use all the available GPUs with DataParallel
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    return model

def main():
    batch_size = 16
    epochs = 3
    lossfunction = torch.nn.BCEWithLogitsLoss()
    num_workers = multiprocessing.cpu_count()
    lr = 0.001
    weight_decay = 1e-5
    kfolds = 5
    n_model_outputs = 2
    verbose = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_time_frames = 1024 # Required by/due to the ASTModel pretraining
    n_mels = 128
    hop_length = 512
    n_fft = 2048,
    scale_melbands=False
    classification_threshold = 0.5

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, n_mels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    # Only used for limited dataset during verification run
    limit = 42

    logger_factory = LoggerFactory(
        logger_type=Logger, 
        logger_args=(LogFormatter(),)
    )
    logger = logger_factory.create_logger()

    worker = Binworker(
        pool_ref=multiprocessing.Pool,
        n_processes=num_workers,
        timeout_seconds=None
    )

    model_provider = DefaultModelProvider(
        model_ref=instantiate_model,
        model_args=(n_model_outputs, n_mels, n_time_frames, device),
        model_kwargs={},
        logger_factory=logger_factory,
        verbose=verbose
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

    complete_tensordataset = TensorAudioDataset(
        dataset=clipped_dataset,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor
    )

    complete_dataset_provider = BasicDatasetProvider(dataset=complete_tensordataset)

    verification_dataset_provider = VerificationDatasetProvider(
        clipped_dataset=clipped_dataset,
        limit=limit,
        balancer=DatasetBalancer(
            dataset=clipped_dataset,
            logger_factory=logger_factory,
            worker=worker,
            verbose=verbose
        ),
        randomize=True,
        balanced=True,
        feature_accessor=feature_accessor,
        label_accessor=label_accessor
    )

    dataset_verifier = BinaryTensorDatasetVerifier(
        logger_factory=logger_factory,
        verbose=verbose
    )

    verification_tracker = Tracker(
        logger_factory=logger_factory,
        client = SheetClient(
            logger_factory=logger_factory, 
            spreadsheet_key="1qT3gS0brhu2wj59cyeZYP3AywGErROJCqR2wYks6Hcw", 
            sheet_id=1339590295
        )
    )

    complete_tracker = Tracker(
        logger_factory=logger_factory,
        client = SheetClient(
            logger_factory=logger_factory, 
            spreadsheet_key=config.SPREADSHEET_ID, 
            sheet_id=config.SHEET_ID
        )
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
        optimizer_type=torch.optim.Adamax,
        optimizer_args=(),
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay}
    )

    trainer = Trainer(
        logger_factory=logger_factory,
        optimizer_provider=optimizer_provider,
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

    metric_computer = BinaryMetricComputer(
        logger_factory=logger_factory,
        class_dict=clipped_dataset.classes(),
        threshold=classification_threshold
    )

    saver = Saver(logger_factory=logger_factory)

    #### Perform "verificaiton" run to check that everything is working as expected.
    cv = CrossEvaluator(
        model_provider=model_provider,
        logger_factory=logger_factory,
        dataset_provider=verification_dataset_provider,
        dataset_verifier=dataset_verifier,
        tracker=verification_tracker,
        folder=folder,
        trainer=trainer,
        evaluator=evaluator,
        metric_computer=metric_computer,
        saver=saver
    )
    cv.kfoldcv()

    logger.log("Verification run complete!")
    logger.log(f"Beginning {kfolds}-fold cross evaluation...")
    #### Perform the proper training run
    cv = CrossEvaluator(
        model_provider=model_provider,
        logger_factory=logger_factory,
        dataset_provider=complete_dataset_provider,
        dataset_verifier=dataset_verifier,
        tracker=complete_tracker,
        folder=folder,
        trainer=trainer,
        evaluator=evaluator,
        metric_computer=metric_computer,
        saver=saver
    )
    cv.kfoldcv()

if __name__ == "__main__":
    main()