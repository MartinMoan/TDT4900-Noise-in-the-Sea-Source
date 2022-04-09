#!/usr/bin/env
import abc
import sys, pathlib
from typing import Iterable, Mapping, Union, Collection
import warnings

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score, recall_score
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

class IMetricComputer(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def metrics(self) -> Iterable[callable]:
        """Return a list of """
        raise NotImplementedError

    def __call__(self, truth: torch.Tensor, preds: torch.Tensor) -> Mapping[str, Union[list, float, str]]:
        """Compute metrics given the model output (preds) and the optimal model output (truth)

        Args:
            truth (torch.Tensor): The optimal/correct model output.
            preds (torch.Tensor): The actual model output.

        Returns:
            Mapping[str, float]: a dict of {metric_name: metric_value} mappings
        """
        print("Computing metrics...")
        print(f"Truth matrix shape: {truth.shape}")
        print(f"Prediction matrix shape: {preds.shape}")
        if len(self.metrics) == 0:
            raise Exception(f"{self.__class__.__name__} does not have any registered metric functions (self.metrics) returned iterable with length 0")

        num_failed = 0
        exceptions = []
        results = {}
        for func in self.metrics:
            result = f"{func.__name__} did not compute"
            try:
                result = func(truth, preds)
            except Exception as ex: 
                result = f"{func.__name__} failed to compute with the following exception: {str(ex)}"
                exceptions.append(ex)
                num_failed += 1

            results[func.__name__] = result
            
        if num_failed == len(self.metrics) and len(self.metrics) != 0:
            caught_exceptions = "\n".join([str(ex) for ex in exceptions])
            raise Exception(f"{self.__class__.__name__}: No metrics could be computed.\n{caught_exceptions}")
        print("Metric computation done!")
        print(results)
        return results

class BinaryMetricComputer(IMetricComputer):
    def __init__(self, class_dict, threshold: float = 0.5):
        self._threshold = threshold
        self._class_dict = class_dict

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

if __name__ == "__main__":
    from rich import print
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.datasets import make_multilabel_classification
    from sklearn.multioutput import MultiOutputClassifier
    import numpy as np
    metrics = BinaryMetricComputer({"Anthropogenic": 0, "Biogenic": 1})
    X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=2, random_state=0)
    inner_clf = LogisticRegression(solver="liblinear", random_state=0)
    clf = MultiOutputClassifier(inner_clf).fit(X, y)
    probs = clf.predict_proba(X)

    y_score = np.transpose([y_pred[:, 1] for y_pred in probs]) # the by-label probabilities per sample. E.g. y_score[14, 3] tells us the probability of sample 14 to have a positive instance of label/class 3
    truth = torch.tensor(y, dtype=torch.float32, requires_grad=False)
    preds = torch.tensor(y_score, dtype=torch.float32, requires_grad=False)
    print(metrics(truth, preds))
    