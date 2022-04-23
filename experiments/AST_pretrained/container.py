#!/usr/bin/env python3
from dataclasses import dataclass
import sys
import pathlib

from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
import git
import torch

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from logger import Logger
from modelprovider import DefaultModelProvider

def instantiate_model(n_model_outputs: int, nmels: int, n_time_frames: int, device: str):
    model = ASTWrapper(
        activation_func = None, # Because loss function is BCEWithLogitsLoss which includes sigmoid activation.
        label_dim=n_model_outputs,
        fstride=10, 
        tstride=10, 
        input_fdim=nmels, 
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

@dataclass
class CustomConfiguration:
    n_model_outputs: int = 2
    nmels: int = 128
    hop_length: int = 512
    n_time_frames: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
    clip_overlap_samples = int(clip_length_samples * 0.25)

    lr: float = 0.001
    weight_decay = 1e-5
    epochs: int = 3
    batch_size: int = 16
    num_workers: int = 8

    verbose: bool = True

    loss_ref = torch.nn.BCEWithLogitsLoss
    optimizer_ref = torch.optim.Adamax

    kfolds: int = 8

class AstContainer(containers.DeclarativeContainer):
    configuration = providers.Singleton(
        CustomConfiguration
    )

    logger = providers.Singleton(Logger)

@inject
def main(configuration: CustomConfiguration = Provide[AstContainer.configuration]):
    print(configuration)

if __name__ == "__main__":
    container = AstContainer()
    container.init_resources()
    container.wire(modules=[__name__])

    main()