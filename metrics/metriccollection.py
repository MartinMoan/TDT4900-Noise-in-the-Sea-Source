#!/usr/bin/env python3
import warnings
import ast
from copy import copy
import sys
import pathlib
from typing import Iterable, Optional, Dict, Any, List, Union, Mapping, Tuple

import git
import torch
import torch.utils.data
import torchmetrics
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

class Average(torchmetrics.Metric):
    def __init__(self, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_examples", default=torch.tensor(1.0), dist_reduce_fx="sum") # default 1 to avoid divide by 0 exception

    def update(self, batch_accuracy: torch.Tensor, *args, **kwargs) -> None:
        self.sum += batch_accuracy
        self.num_examples += torch.tensor(1.0).type_as(batch_accuracy)

    def compute(self) -> torch.Tensor:
        MIN = torch.tensor(1.0).type_as(self.num_examples)
        if self.num_examples > MIN:
            return torch.div(self.sum, self.num_examples - MIN)
        else:
            return torch.div(self.sum, self.num_examples)

class ExampleCounter(torchmetrics.Metric):
    def __init__(self, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0),  dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        self.sum += preds.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum

class Support(torchmetrics.Metric):
    def __init__(self, num_classes: int, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.num_classes = num_classes
        self.add_state("num_examples", default=torch.tensor(1.0), dist_reduce_fx="sum") # default 1 to avoid divide by 0 error
        for c in range(self.num_classes):
            self.add_state(f"num_target_class{c}", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape, f"preds and target have different shapes preds: {preds.shape} target: {target.shape}"
        assert preds.dim() == 2, f"input has incorrect number of dimensions, expected 2D torch.Tensor but received {preds.dim()}D Tensors"
        assert preds.shape[1] == self.num_classes, f"input has unexpected shape, expected (batch_size, num_classes={self.num_classes}) but received tensor with shape {preds.shape}"
        uq = torch.unique(target).data.cpu().numpy()
        assert set(uq).issubset(set([0, 1])), f"Target is not binary integer tensor, expected unique values [0, 1] but found unique values {uq} in tensor {target}"
        
        self.num_examples = self.num_examples.type_as(target).to(target.device)
        self.num_examples += target.shape[0]
        
        counts = torch.sum(target, dim=0).type_as(target).to(target.device)
        for c in range(self.num_classes):
            count = counts[c]
            state = getattr(self, f"num_target_class{c}")
            state += count

    def compute(self) -> torch.Tensor:
        # return torch.tensor([torch.div(getattr(self, f"num_target_class{c}"), self.num_examples) for c in range(self.num_classes)]).float().to(self.device)
        return torch.tensor([getattr(self, f"num_target_class{c}") for c in range(self.num_classes)]).float().to(self.device)

class GliderMetrics(torchmetrics.MetricCollection):
    def __init__(self, num_classes: int, *args, missing_reset: str = "warn", class_names: Optional[Union[List[str], np.ndarray, Tuple[str, ...]]] = None,  **kwargs):
        if class_names is None:
            class_names = [f"class{i}" for i in range(num_classes)]
        
        if missing_reset not in ["warn", "fail", None]:
            raise ValueError(f"Argument 'missing_reset' has incorrect value, expected string with value 'warn', 'fail' or None but received {type(missing_reset)} with value {missing_reset}")

        if not isinstance(class_names, (list, np.ndarray, tuple)):
            raise TypeError(f"Argument 'class_names' has incorrect type, expected 'list' or 'np.ndarray' but received object with type {type(class_names)}")

        if len(class_names) != num_classes:
            raise ValueError(f"Argument 'class_names' has incorrect number of elements, must have lenght equal to 'num_classes' argument")
            
        averaging_techniques = ["weighted"]
        classes = [torchmetrics.Accuracy, torchmetrics.AUROC, torchmetrics.Precision, torchmetrics.Recall, torchmetrics.AveragePrecision, torchmetrics.F1Score]
        metrics = {}
        for average in averaging_techniques:
            for metricClass in classes:
                attributes = {
                    "name": metricClass.__name__,
                    "average": average
                }
                name = str(attributes)
                metrics[name] = metricClass(num_classes=num_classes, average=average)

        supportname = dict(name=Support.__name__, average=None)
        metrics[str(supportname)] = Support(num_classes=num_classes) # The percentage of class instances per class (e.g. [90% class0, 80% class1, 10% class2, ...])

        examplesname = dict(name=ExampleCounter.__name__, average=None)
        metrics[str(examplesname)] = ExampleCounter()

        super().__init__(metrics, *args, compute_groups=False, **kwargs) # Need compute groups False because grouping causes metric.update() not to be called internally for some metrics when wrapped in this GliderMetric object
        self.num_classes = num_classes
        self.class_names = class_names
        self._compute_called_since_last_reset = False
        self.missing_reset_action = missing_reset

    def _cast_metrics_to_tensor(self, metrics: Mapping[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Mapping[str, torch.Tensor]:
        output = {}
        for name, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                output[name] = torch.Tensor(value)
            else:
                output[name] = value
        return output

    def _expand_metrics_per_class(self, metrics: Mapping[str, torch.Tensor]) -> List[Mapping[str, Union[Mapping[str, Any], torch.Tensor]]]:
        """
        Returns:
            Iterable[Mapping[str, Union[Mapping[str, Any], torch.Tensor]]]: f.ex.:
            [
                {
                    "value": torch.Tensor([0.8431], dtype=...),
                    "attributes": {
                        "class": (str) "class0" | "class1" | "CustomClassName1" | ...,
                        "step": (str) "train" | "val" | "test",
                        "name": (str) "Accuracy" | "F1Score" ...,
                        "average": "weighted" | "micro" | "macro" ...,
                        ...
                    }
                }
            ]
        """
        output = []
        for key, value in metrics.items():
            attributes = dict(ast.literal_eval(key))
            if value.dim() == 0:
                output.append(dict(value=value, attributes=attributes))
                continue
            
            if value.dim() != 1:
                raise Exception(f"Metric '{key}' is not 1D Tensor, but {value.dim()}D Tensor: {repr(value)}")
            if value.shape[0] == len(self.class_names):
                # Metric is computed per-class, expand metric into self.num_classes individual metrics
                # This is to enable logging of per-class computed metrics, as PytorchLightning raises an exception whenever a the LightningModule mehtod self.log() is called with a tensor with multiple values
                for cls in range(self.num_classes):
                    metric_value = value[cls]
                    attrs = copy(attributes)
                    attrs["classname"] = self.class_names[cls]
                    output.append(dict(value=metric_value, attributes=attrs))
        return output

    def _rename(self, step: Optional[str] = None, name: str = "NamelessMetric", average: Optional[str] = None, classname: Optional[str] = None) -> str:
        stepstr = f"{step.strip().capitalize()} " if step is not None else ""
        namestr = f"{name}" if name is not None else ""
        classnamestr = f" {classname}" if classname is not None else ""
        averagestr = f" (average={average})" if average is not None else ""
        return f"{stepstr}{namestr}{classnamestr}{averagestr}"

    def _rename_metrics(self, metrics: Iterable[Mapping], step: Optional[str] = None) -> Dict[str, torch.Tensor]:
        output = {}
        for metric in metrics:
            value = metric.get("value") or -1
            attributes = metric.get("attributes") or {}
            output[self._rename(step=step, **attributes)] = value
        return output

    def _clean_metric_output(self, metrics: Mapping[str, Union[torch.Tensor, List[torch.Tensor]]], step: Optional[str] = None) -> Mapping[str, torch.Tensor]:
        metrics = self._cast_metrics_to_tensor(metrics)
        metrics = self._expand_metrics_per_class(metrics)
        metrics = self._rename_metrics(metrics, step=step)
        return metrics

    def __call__(self, step: str, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        cn = self.__class__.__name__
        warnings.wand(f"{cn}.__call__() was called. This will likely cause incorrect results. In stead of calling {cn} directly, make individual calls to {cn}.update(), {cn}.compute() and {cn}.reset(), as this will ensure state is maintained correctly and metrics are correctly computed.")
        self.reset()
        self.update(preds, target)
        return self.compute(step)

    def compute(self, step: str = None) -> Dict[str, Any]:
        metrics = super().compute()
        self._compute_called_since_last_reset = True
        return self._clean_metric_output(metrics, step=step)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self._compute_called_since_last_reset:
            cn = self.__class__.__name__
            msg = f"{cn}.update() was called after {cn}.compute(), but metrics was not reset since last compute call. This will cause previously computed metrics to accumulate with new data, and likely yield incorrect results."
            if self.missing_reset_action == "warn":
                warnings.warn(msg)
            elif self.missing_reset_action == "fail":
                raise Exception(msg)
                
        return super().update(preds, target)
    
    def reset(self) -> None:
        super().reset()
        self._compute_called_since_last_reset = False

def cls(preds, threshold=0.5):
    classified = torch.clone(preds)
    classified[classified > threshold] = 1
    classified[classified <= threshold] = 0
    return classified

def support(target):
    return torch.sum(target, dim=0)

def confusion(target, classes):
    TP = torch.sum(torch.logical_and(classes.bool(), target.bool()).int(), dim=0)
    FN = torch.sum(torch.logical_and(torch.logical_not(classes.bool()), target.bool()).int(), dim=0)
    TN = torch.sum(torch.logical_and(torch.logical_not(classes.bool()), torch.logical_not(target.bool())).int(), dim=0)
    FP = torch.sum(torch.logical_and(classes.bool(), torch.logical_not(target.bool())).int(), dim=0)
    return dict(TP=TP, FN=FN, TN=TN, FP=FP)

def precision(target, preds, threshold=0.5):
    # TP / (TP + FP)
    c = cls(preds, threshold=threshold)
    conf = confusion(target, c)
    TP, FP = conf["TP"], conf["FP"]
    return torch.div(TP, (TP + FP), out=torch.zeros_like(TP).float()).nan_to_num(0.0)

def recall(target, preds, threshold=0.5):
    # TP / (TP + FN)
    c = cls(preds, threshold=threshold)
    conf = confusion(target, c)
    TP, FN = conf["TP"], conf["FN"]
    return torch.div(TP, (TP + FN), out=torch.zeros_like(TP).float()).nan_to_num(0.0)

def precision_recall_curve(targets, preds, n_thresholds=10):
    # thresholds = np.linspace(0, 1, num=n_thresholds)
    thresholds = torch.sort(preds, dim=0)
    precisions = np.array([precision(targets, preds, threshold=threshold) for threshold in thresholds])
    precisions = np.array([[prec[c] for prec in precisions] for c in range(targets.shape[1])])
    recalls = np.array([recall(targets, preds, threshold=threshold) for threshold in thresholds])
    recalls = np.array([[rec[c] for rec in recalls] for c in range(targets.shape[1])])
    return precisions, recalls, thresholds

if __name__ == "__main__":
    import pytorch_lightning as pl
    pl.seed_everything(42)
    from rich import print
    num_classes = 2
    size = 10
    target = torch.randint(0, 2, size=(size, num_classes)).int()
    preds = torch.sigmoid((torch.rand((size, num_classes)).float() * 2) - 1)
    print(target)
    print(preds)
    g = GliderMetrics(num_classes=num_classes, class_names=["Anthropogenic", "Biophonic"], missing_reset="warn")

    g.update(preds, target)
    print(g.compute(step="val"))