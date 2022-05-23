#!/usr/bin/env python3
import ast
from copy import copy
import sys
import pathlib
from typing import Iterable, Optional, Dict, Any, List, Union, Mapping

import git
import torch
import torch.utils.data
import torchmetrics
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

class Support(torchmetrics.Metric):
    def __init__(self, num_classes: int, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.num_classes = num_classes
        self.add_state("num_examples", default=torch.tensor(1), dist_reduce_fx="sum") # default 1 to avoid divide by 0 error
        for c in range(self.num_classes):
            self.add_state(f"num_target_class{c}", default=torch.tensor(0), dist_reduce_fx="sum")

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
    def __init__(self, num_classes: int, *args, class_names: Optional[Iterable[str]] = None,  **kwargs):
        if class_names is None:
            class_names = [f"class{i}" for i in range(num_classes)]
        
        if not isinstance(class_names, (list, np.ndarray)):
            raise TypeError(f"Argument 'class_names' has incorrect type, expected 'list' or 'np.ndarray' but received object with type {type(class_names)}")

        if len(class_names) != num_classes:
            raise ValueError(f"Argument 'class_names' has incorrect number of elements, must have lenght equal to 'num_classes' argument")
            
        averaging_techniques = ["micro", "macro", "weighted", None]
        # averaging_techniques = ["weighted"]
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
        super().__init__(metrics, *args, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names

    def _cast_metrics_to_tensor(self, metrics: Mapping[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Mapping[str, torch.Tensor]:
        output = {}
        for name, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                output[name] = torch.Tensor(value)
            else:
                output[name] = value
        return output

    def _expand_metrics_per_class(self, metrics: Mapping[str, torch.Tensor]) -> Iterable[Mapping[str, Union[Mapping[str, Any], torch.Tensor]]]:
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
                    output.append(dict(value=torch.Tensor(metric_value), attributes=attrs))
        return output

    def _rename(self, step: Optional[str] = None, name: str = "NamelessMetric", average: Optional[str] = None, classname: Optional[str] = None) -> str:
        stepstr = f"{step.strip().capitalize()} " if step is not None else ""
        namestr = f"{name}" if name is not None else ""
        classnamestr = f" {classname}" if classname is not None else ""
        averagestr = f" (average={average})" if average is not None else ""
        return f"{stepstr}{namestr}{classnamestr}{averagestr}"

    def _rename_metrics(self, metrics: Iterable[Mapping], step: Optional[str] = None) -> Mapping[str, torch.Tensor]:
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

    def __call__(self, *args, **kwargs):
        metrics = super().__call__(*args, **kwargs)
        return self._clean_metric_output(metrics, step=kwargs.get("step") or None)

    def compute(self, step: str = None) -> Dict[str, Any]:
        metrics = super().compute()
        return self._clean_metric_output(metrics, step=step)

if __name__ == "__main__":
    from rich import print
    num_classes = 2
    size = 1000
    g = GliderMetrics(num_classes=num_classes, class_names=["Anthropogenic", "Biophonic"])
    
    target = torch.randint(0, 2, (size, num_classes)).int()
    preds = torch.rand((size, num_classes)).float()

    print(g)
    g.update(preds, target)
    print(g.compute(step="val"))