#!/usr/bin/env python3
import pathlib
import sys
import os
from datetime import datetime
import random
from typing import Mapping, Optional, Union, Iterable

import pandas as pd
from rich import print
import git
import wandb

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ITracker, ILoggerFactory, ITabularLogger

class WandbTracker(ITracker):
    def __init__(
        self, 
        group: str = None,
        job_type: str = None,
        name: str = None, 
        note: Optional[str] = None, 
        tags: Optional[Iterable[str]] = None, 
        **kwargs) -> None:

        uname = os.uname()
        self.infrastructure = {
            "nodename": uname.nodename,
            "release": uname.release,
            "sysname": uname.sysname,
            "version": uname.version,
            "machine": uname.machine
        }
        configuration = {**self.infrastructure, **kwargs}

        self.run = wandb.init(
            name=name,
            project=os.environ.get("WANDB_PROJECT", "MISSING_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY", "MISSING_ENTITY"),
            notes=note,
            tags=tags,
            save_code=True,
            group=group,
            job_type=job_type,
            dir=str(config.HOME_PROJECT_DIR.absolute()),
            config=configuration
        )
    
    def track(self, trackables: Mapping[str, any], **kwargs) -> None:
        self.run.log({**trackables, **kwargs})

class SummableDict:
    def _isnumber(value: any) -> bool:
        return isinstance(value, (int, float))

    def _add(d1: Mapping[str, any], d2: Mapping[str, any]):
        if SummableDict._isnumber(d1) and SummableDict._isnumber(d2):
            return d1 + d2
        
        output = {}
        for key in d1.keys():
            if key in d2.keys():
                output[key] = SummableDict._add(d1[key], d2[key])
            else:
                output[key] = d1[key]
        
        for key in d2.keys():
            if key not in output.keys():
                if key in d1.keys():
                    output[key] = SummableDict._add(d1[key], d2[key])
                else:
                    output[key] = d2[key]
        return output

    def add(d1: Mapping[str, any], d2: Mapping[str, any]):
        output = SummableDict._add(d1, d2)
        return output

    def _div(value, divisor):
        if SummableDict._isnumber(value):
            return value / divisor
        
        if not isinstance(value, dict):
            raise TypeError

        output = {}
        for key in value.keys():
            output[key] = SummableDict._div(value[key], divisor)
        return output

    def div(dict: Mapping[str, any], divisor: Union[float, int]):
        if not SummableDict._isnumber(divisor):
            raise TypeError
        
        return SummableDict._div(dict, divisor)
            
if __name__ == "__main__":
    kfolds = 5
    epochs = 100
    batches = 5
    batch_size = 16

    model_name = "TestModel"
    now = datetime.now()
    started_at = now.strftime("%H:%M %d. %b %Y")

    tracker = WandbTracker(
        name=f"{model_name} - {started_at}", 
        note="A quick test with smaller learning_rate",
    )

    CUM_METRICS = {}
    for fold in range(kfolds):
        # train
        
        train_batches = int(batch_size * (batches - 1))
        eval_batches = int(batch_size)
        
        for epoch in range(epochs):
            cumu_loss = 0
            
            for batch in range(train_batches):
                batch_loss = random.random() * (1 / (epoch + 1))
                tracker.track(
                    {
                        "batch_loss": batch_loss, 
                        "epoch": epoch, 
                        "batch": batch, 
                        "fold": fold
                    }
                )

                cumu_loss += batch_loss
            
            avg_epoch_loss = cumu_loss / batches
            tracker.track(
                {
                    "loss": avg_epoch_loss, 
                    "epoch": epoch, 
                    "fold": fold
                }
            )

        # eval
        all_ys, all_preds = [], []
        for testbatch in range(eval_batches):
            pass

        metrics = {
            "accuracy": random.random(),
            "f1": {
                "Anthropogenic": random.random(),
                "Biophonic": random.random()
            },
            "precision": {
                "Anthropogenic": random.random(),
                "Biophonic": random.random()
            },
            "recall": {
                "Anthropogenic": random.random(),
                "Biophonic": random.random()
            },
            "roc_auc": {
                "Anthropogenic": random.random(),
                "Biophonic": random.random()
            },
        }

        tracker.track({"fold_metrics": metrics, "fold": fold})

        CUM_METRICS = SummableDict.add(CUM_METRICS, metrics)
    
    CUM_METRICS = SummableDict.div(CUM_METRICS, kfolds)
    tracker.track({"eval": CUM_METRICS})