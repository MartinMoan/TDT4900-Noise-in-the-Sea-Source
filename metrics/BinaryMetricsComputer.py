#!/usr/bin/env python3
import pathlib
import sys
from typing import Iterable, Mapping, Union
import warnings

from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score, recall_score
import torch
import numpy as np
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from tracking.logger import Logger
from interfaces.IMetricComputer import IMetricComputer
from interfaces.ILogger import ILogger

class BinaryMetricComputer(IMetricComputer):
    def __init__(self, class_dict, threshold: float = 0.5, logger: ILogger = Logger()):
        self._threshold = threshold
        self._class_dict = class_dict
        self._logger = logger

    def _apply_threshold(self, multiplabel_indicator: torch.Tensor) -> torch.Tensor:
        preds = multiplabel_indicator
        positives = preds > self._threshold
        preds[positives] = 1
        preds[~positives] = 0
        return preds

    def _check_is_2d(self, tensor_name: str, tensor: torch.Tensor):
        truth_value_error = ValueError(f"BinaryMetricComputer received {tensor_name} matrix with invalid dimension. Expeted 2-dimensional matrix of shape (batch_size, 2) but received {tensor.ndim}-dimensional object with shape {tensor.shape}")
        if tensor.ndim != 2:
            raise truth_value_error
        batch_size, num_values = tensor.shape
        if num_values != 2:
            raise truth_value_error

    def _iterable_scores_to_dict(self, scores):
        return {key: scores[self._class_dict[key]] for key in self._class_dict.keys()}

    def __call__(self, truth: torch.Tensor, preds: torch.Tensor) -> Mapping[str, Union[list, float, str]]:
        """
        Args:
            truth (torch.Tensor): Truth matrix with shape (batch_size, 2)
            preds (torch.Tensor): Prediction matrix with shape (batch_size, 2)

        Returns:
            Mapping[str, Union[list, float, str]]: Mapping of metric_name: metric_value(s)/error_message
        """
        self._logger.log("Beginning metric computation...")
        self._check_is_2d("truth", truth)
        self._check_is_2d("preds", preds)

        if config.ENV == "dev":
            # Set truth values to ensure that all metrics actually can be computed. F.ex. roc_auc is not defined if only one class present in truth matrix.
            warnings.warn("BinaryMetricComputer detected ENV=dev; will entirely overwrite the truth matrix to ensure balanced classes/labels.")
            quarter = int(truth.shape[0] / 4)
            truth[:quarter, 0] = 0
            truth[:quarter, 1] = 0

            truth[quarter:int(2 * quarter), 0] = 1
            truth[quarter:int(2 * quarter), 1] = 0
            
            truth[int(2*quarter):int(3*quarter), 0] = 0
            truth[int(2*quarter):int(3*quarter), 1] = 1
            
            truth[int(3*quarter):, 0] = 1
            truth[int(3*quarter):, 1] = 1

            preds = torch.rand(truth.shape)

        if not set(np.unique(truth)).issubset(set([0.0, 1.0])):
            raise ValueError(f"{self.__class__.__name__} received truth matrix with invalid values. Expected 2-dimensional label indicator with shape (batch_size, 2). truth matrix contains values not in the set {set([0.0, 0.1])}")
        
        return super().__call__(truth, preds)

    @property
    def metrics(self) -> Iterable[callable]:
        return [self.accuracy, self.precision, self.f1, self.roc_auc, self.recall]
    
    def accuracy(self, truth: torch.Tensor, preds: torch.Tensor) -> float:
        return accuracy_score(truth, self._apply_threshold(preds))

    def roc_auc(self, truth: torch.Tensor, preds: torch.Tensor) -> Iterable[float]:
        return self._iterable_scores_to_dict(roc_auc_score(truth, preds, average=None))

    def precision(self, truth: torch.Tensor, preds: torch.Tensor) -> Iterable[float]:
        return self._iterable_scores_to_dict(precision_score(truth, self._apply_threshold(preds), average=None, zero_division=0))

    def f1(self, truth: torch.Tensor, preds: torch.Tensor) -> Iterable[float]:
        return self._iterable_scores_to_dict(f1_score(truth, preds, average=None, zero_division=0))

    def recall(self, truth: torch.Tensor, preds: torch.Tensor) -> Iterable[float]:
        return self._iterable_scores_to_dict(recall_score(truth, self._apply_threshold(preds), average=None, zero_division=0))

    def logger(self) -> ILogger:
        return self._logger