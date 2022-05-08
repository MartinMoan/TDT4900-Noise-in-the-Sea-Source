#!/usr/bin/env python3 
import argparse
import sys
import pathlib
from typing import Tuple, Iterable

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

from interfaces import ILoggerFactory, ITensorAudioDataset
from tracking.logger import Logger, LogFormatter
from tracking.loggerfactory import LoggerFactory

from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.balancing import DatasetBalancer, CachedDatasetBalancer
from datasets.binjob import Binworker
from datasets.tensordataset import BinaryLabelAccessor, MelSpectrogramFeatureAccessor, TensorAudioDataset

from models.AST.AST import ASTModel
from models.AST.ASTWrapper import ASTWrapper
import torchmetrics

def create_tensorset(
    logger_factory: ILoggerFactory, 
    nfft: int, 
    nmels: int, 
    hop_length: int, 
    clip_duration_seconds: float, 
    clip_overlap_seconds: float, 
    force_recache: bool = False) -> Tuple[ITensorAudioDataset, Iterable[int], Iterable[int]]:

    clips = CachedClippedDataset(
        logger_factory=logger_factory,
        worker=Binworker(),
        clip_duration_seconds=clip_duration_seconds,
        clip_overlap_seconds=clip_overlap_seconds,
        force_recache=force_recache
    )

    label_accessor = BinaryLabelAccessor()
    feature_accessor = MelSpectrogramFeatureAccessor(
        logger_factory=logger_factory,
        n_mels=nmels,
        n_fft=nfft,
        hop_length=hop_length,
        scale_melbands=False,
        verbose=True
    )

    tensorset = TensorAudioDataset(
        dataset=clips,
        label_accessor=label_accessor,
        feature_accessor=feature_accessor,
        logger_factory=logger_factory
    )

    balancer = CachedDatasetBalancer(
        dataset=clips,
        logger_factory=logger_factory,
        worker=Binworker(),
        verbose=True,
        force_recache=force_recache
    )

    eval_only_indeces = balancer.eval_only_indeces()
    train_indeces = balancer.train_indeces()

    return tensorset, eval_only_indeces, train_indeces

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nmels", type=int, required=True)
    parser.add_argument("-nfft", type=int, required=True)
    parser.add_argument("-hop_length", type=int, required=True)
    parser.add_argument("-clip_duration_seconds", type=float, required=True)
    parser.add_argument("-clip_overlap_seconds", type=float, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = getargs()
    logger_factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(),
        logger_kwargs=dict(logformatter=LogFormatter())
    )

    tensorset, eval_only, train = create_tensorset(
        logger_factory=logger_factory,
        nfft=args.nfft,
        nmels=args.nmels,
        hop_length=args.hop_length,
        clip_duration_seconds=args.clip_duration_seconds,
        clip_overlap_seconds=args.clip_overlap_seconds,
        force_recache=False
    )

    import numpy as np
    np.random.shuffle(train)

    accuracy = torchmetrics.Accuracy(num_classes=2)
    auc = torchmetrics.AUC(reorder=True)
    aucroc = torchmetrics.AUROC(num_classes=2)
    precision = torchmetrics.Precision(num_classes=2)
    recall = torchmetrics.Recall(num_classes=2)
    average_precision = torchmetrics.AveragePrecision(num_classes=2)
    f1 = torchmetrics.F1Score(num_classes=2)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True)

    sr = 128000
    fdim = args.nmels
    tdim = int((args.clip_duration_seconds * sr / args.hop_length) + 1)
    import torch
    ast = ASTWrapper(
        logger_factory=logger_factory,
        activation_func=torch.nn.Sigmoid(),
        label_dim=2, 
        fstride=16, 
        tstride=16, 
        input_fdim=fdim, 
        input_tdim=tdim, 
        imagenet_pretrain=True, 
        audioset_pretrain=False, 
        model_size="tiny224", 
        verbose=True
    )
    n = 50
    for i in range(min(n, len(train))):
        # batch_size, 1, n_mel_bands, n_time_frames
        X, Y = tensorset[train[i]]
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(0)
        Yhat = ast(X)
        print(i, min(n, len(train)), Y, Yhat)
        
        accuracy.update(Yhat, Y.int())
        auc.update(Yhat, Y)
        aucroc.update(Yhat, Y.int())
        precision.update(Yhat, Y.int())
        recall.update(Yhat, Y.int())
        average_precision.update(Yhat, Y)
        f1.update(Yhat, Y.int())
        confusion_matrix.update(Yhat, Y.int())

    print("accuracy:", accuracy.compute())
    print("auc:", auc.compute())
    print("aucroc:", aucroc.compute())
    print("precision:", precision.compute())
    print("recall:", recall.compute())
    print("average_precision:", average_precision.compute())
    print("f1:", f1.compute())
    print("confusion_matrix:", confusion_matrix.compute())