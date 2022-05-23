#!/usr/bin/env python3
from typing import Optional, Dict, Any
import pathlib
import sys

import git
import numpy as np
import torch
import torchmetrics
from torchmetrics import Metric

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

def is_int_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]

class MeanAveragePrecision(Metric):
    def __init__(self, num_classes: int, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.num_classes = num_classes
        self.internals = torchmetrics.AveragePrecision(num_classes=num_classes, average=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update internal state needed to compute mAP (mean Average Precision)

        Args:
            preds (torch.Tensor): "binary" int Tensor (only containing values 0 and/or 1) with shape (B=batch_size * N_batches, C=num_classes)
            target (torch.Tensor): float tensor with same shape as preds (B, C)
        """
        assert isinstance(target, torch.Tensor), f"target has incorrect type, expected torch.Tensor but received object with type {type(target)}"
        assert isinstance(preds, torch.Tensor), f"preds has incorrect type, expected torch.Tensor but received object with type {type(preds)}"
        assert preds.dim() == 2, f"preds have incorrect dimensions, expected 2D imput but found {preds.dim()}"
        assert target.dim() == 2, f"target have incorrect dimensions, expected 2D input but found {target.dim()}"
        assert preds.shape[0] == target.shape[0], f"target and preds have different shapes, equal shape (batches, num_classes) but received target with shape {target.shape} and preds with shape {preds.shape}"
        B = preds.shape[0]
        assert preds.shape[1] == target.shape[1], f"target and preds have different shapes, equal shape (batches, num_classes) but received target with shape {target.shape} and preds with shape {preds.shape}"
        assert preds.shape[1] == self.num_classes, f"target and preds have unexpected number of classes, expected num_classes to be {self.num_classes} but second axis of target and preds has {preds.shape[1]} elements"
        assert is_int_tensor(target), f"target values has incorrect datatype, expected torch.int tensor but received tensor with dtype {target.dtype}"
        assert torch.is_floating_point(preds), f"preds values has incorrect datatype, expected float tensor but received tensor with dtype {preds.dtype}"
        uq = torch.unique(target).data.cpu().numpy()
        assert set(uq).issubset(set([0, 1])), f"Target is not binary integer tensor, expected unique values [0, 1] but found unique values {uq} in tensor {target}"
        self.internals.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the mAP of all previusly seen predictions and targets

        Returns:
            torch.Tensor: tensor with a single float value, representing the mAP of all seen predictions and targets.
        """
        average_precision_values = self.internals.compute()
        return torch.mean(torch.Tensor(average_precision_values))