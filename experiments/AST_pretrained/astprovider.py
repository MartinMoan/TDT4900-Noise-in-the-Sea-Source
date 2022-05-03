#!/usr/bin/env python3
import sys
import pathlib
from typing import Mapping

import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import ILoggerFactory, IModelProvider

from models.AST.ASTWrapper import ASTWrapper

class AstModelProvider(IModelProvider):
    def __init__(
        self,
        logger_factory: ILoggerFactory,
        n_model_outputs: int, 
        n_mels: int,
        hop_length: int,
        clip_length_samples: int,
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
        self.hop_length = hop_length
        self.clip_length_samples = clip_length_samples
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
