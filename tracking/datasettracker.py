#!/usr/bin/env python3
import pathlib
import sys
import os

import numpy as np
from rich import print
import git
import wandb
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from datasets.datamodule import ClippedGliderDataModule

def track_dataset(logger: WandbLogger, datamodule: ClippedGliderDataModule, n_examples: int = 50):
    dataset = datamodule.get_tensor_audio_dataset()

    slurm_procid = int(os.environ.get("SLURM_PROCID", default=-1))
    if slurm_procid != 0 and slurm_procid != -1:
        return

    if n_examples <= 0 or len(dataset) < n_examples:
        return
    
    indeces = np.random.random_integers(0, len(dataset) - 1, n_examples)

    table = wandb.Table(columns=[
        "audio", 
        "spectrogram", 
        "source_class_specific", 
        "clip_classes", 
        "dataset_classes", 
        "tensor_label", 
        "filename", 
        "file_start_time", 
        "file_end_time", 
        "clip_offset_seconds", 
        "clip_duration_seconds",
        "clip_index"
    ])

    for index in range(len(indeces)):
        i = indeces[index]
        clip = dataset.audiodata(i)
        X, Y = dataset[i]
        audio = wandb.Audio(clip.samples, sample_rate=clip.sampling_rate)
        spectrogram = wandb.Image(X, caption=f"{clip.filepath.name}")
        source_class_specific = ", ".join(np.sort(clip.labels.source_class_specific.unique(), axis=0))
        clip_classes = ", ".join(np.sort(clip.labels.source_class.unique(), axis=0))
        dataset_classes = str(dataset.classes())
        tensor_label = str(Y.detach().numpy())
        filename = clip.filepath.name
        datetimeformat = "%c"
        file_start_time = clip.file_start_time.strftime(datetimeformat)
        file_end_time = clip.file_end_time.strftime(datetimeformat)
        clip_offset = clip.clip_offset
        clip_duration = clip.clip_duration

        table.add_data(
            audio, 
            spectrogram, 
            source_class_specific, 
            clip_classes, 
            dataset_classes, 
            tensor_label,
            filename,
            file_start_time,
            file_end_time,
            clip_offset,
            clip_duration,
            i
        )
        print(f"Logging dataset example {index} / {len(indeces)}")
    
    logger.experiment.log({"examples": table})
    logger.experiment.config.update(dict(
        example_shape=dataset.example_shape(),
        label_shape=dataset.label_shape(),
        **datamodule.loggables()
    ))