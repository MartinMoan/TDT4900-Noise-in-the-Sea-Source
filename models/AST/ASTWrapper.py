#!/usr/bin/env python3
import argparse
import pathlib
import sys

import torch
from rich import print
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ILoggerFactory
from AST import ASTModel

class ASTWrapper(torch.nn.Module):
    """
    AST (Audio Spectrogram Transformer) pretraining wrapper. Enables custom activation
    """
    def __init__(
        self, 
        logger_factory: ILoggerFactory,
        activation_func: torch.nn.Module = None, 
        label_dim=2, 
        fstride=10, 
        tstride=10, 
        input_fdim=128, 
        input_tdim=1024, 
        imagenet_pretrain=True, 
        audioset_pretrain=False, 
        model_size='base384', 
        verbose=True) -> None:

        super().__init__()
        self._ast = ASTModel(
            logger_factory=logger_factory,
            label_dim=label_dim, 
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

    def freeze_pretrained_parameters(self) -> None:
        self._ast.freeze_pretrained_params()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Expect X to have shape (batch_size, 1, n_mel_bands, n_time_frames)"""
        X = X.squeeze(dim=1)
        X = self._ast(X)
        if self._activation is not None:
            X = self._activation(X)
        return X 

