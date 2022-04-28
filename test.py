#!/usr/bin/env python3
import os
import random
from datetime import datetime
from rich import print
import time
import sklearn
from sklearn import svm
import wandb

# X, Y = sklearn.datasets.load_iris(return_X_y=True)
# model = svm.SVC()

wandb.init(
    project="testproject", 
    entity="martinmoan",
    notes="This is a note",
    tags=["testrun"]
)

uname = os.uname()
infrastructure = {
    "nodename": uname.nodename,
    "release": uname.release,
    "sysname": uname.sysname,
    "version": uname.version,
    "machine": uname.machine
}

now = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
wandb.run.name = f"TestModel {now}"
wandb.config = {
    "learning_rate": 0.001,
    "something": "with a value",
    "architecture": "No model...",
    "infrastructure": infrastructure
}

for i in range(100):
    print(i)
    to_log = {
        "loss": 1/(i+1),
        "accuracy": random.random(),
        "f1": {
            "Biophonic": random.random(),
            "Anthropogenic": random.random()
        }
    }
    wandb.log(to_log)
