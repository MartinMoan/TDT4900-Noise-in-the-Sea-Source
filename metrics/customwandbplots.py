#!/usr/bin/env python3
from typing import Union, List

import torch
import numpy as np
import wandb

def confusion_matrix(confusion: Union[torch.Tensor, np.ndarray], class_names: List[str] = None, title: str = None):
    if not isinstance(confusion, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Incorrect type for confusion matrix argument, expected torch.Tensor or numpy.ndarray but received type {type(confusion)}")
    if len(confusion.shape) != 2: # Must be 2 dimensional
        raise ValueError(f"Input argument confusion matrix has invalid number of dimensions, expected 2d array but received matrix with dims {len(confusion.shape)}")
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Input argument confusion matrix is not a square matrix, expected shape (n_classes, n_classes) but received matrix with shape {confusion.shape}")
    
    n_classes = confusion.shape[0]
    
    if class_names is not None:
        if not isinstance(class_names, list):
            raise TypeError(f"Input argument class_names has incorrect type, expected python list but received object of type {type(class_names)}")
        if len(class_names) != n_classes:
            raise ValueError(f"Argument class_names has incorrect number of elements, expected to have length equal to confusion argument shape elements (n_classes) {n_classes} but received list with shape {len(class_names)}")
    else:
        class_names = [f"Class{i}" for i in range(n_classes)]

    values = None
    if isinstance(confusion, torch.Tensor):
        values = confusion.detach().numpy()
    else: # is instance of np.ndarray
        values = confusion

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    title = title or ""
    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([class_names[i], class_names[j], values[i, j]])
    
    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields,
        {"title": title},
    )

if __name__ == "__main__":
    import os
    import torchmetrics

    cnf = torchmetrics.ConfusionMatrix(num_classes=2, multilabel=True)
    target = torch.Tensor(
        [
            [0, 1], 
            [0, 1],

            [0, 1],
            [0, 1], 
            [0, 1],

            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],

            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ]
    ).int()
    
    preds = torch.Tensor(
        [
            [0, 1], # c1 TN - c2 TP
            [0, 1], # c1 TN - c2 TP

            [1, 0], # c1 FP - c2 FN
            [1, 0], # c1 FP - c2 FN
            [1, 0], # c1 FP - c2 FN

            [0, 1], # c1 FN - c2 FP
            [0, 1], # c1 FN - c2 FP
            [0, 1], # c1 FN - c2 FP
            [0, 1], # c1 FN - c2 FP

            [1, 0], # c1 TP - c2 TN
            [1, 0], # c1 TP - c2 TN
            [1, 0], # c1 TP - c2 TN
            [1, 0], # c1 TP - c2 TN
            [1, 0], # c1 TP - c2 TN
        ])
    
    # c1 (2 TN, 3 FP, F FN, 5 TP)
    # c2 (5 TN, 4 FP, 3 FN, 2 TP)
    # expect confusion[0] (class 1) to be: [[2, 3], [4, 5]]
    # expect confusion[1] (class 2) to be: [[5, 4], [3, 2]]
    confusion_matrix.update(preds, target)
    confusion = confusion_matrix.compute()
    print(confusion)
    print(confusion[0])
    print(confusion[1])
    # run = wandb.init(project=os.environ.get("WANDB_PROJECT"), entity=os.environ.get("WANDB_ENTITY"))

    # bio = confusion[0]
    # anth = confusion[1]
    # wandb.log({"bio": confusion_matrix(bio, class_names=["not bio", "bio"], title="Biophonic")})
    # wandb.log({"anth": confusion_matrix(anth, class_names=["not anth", "anth"], title="Anthropogenic")})