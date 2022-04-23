#!/usr/bin/env
import sys, pathlib

import torch
import numpy as np
import git
from rich import print
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from metrics import BinaryMetricComputer

if __name__ == "__main__":
    metrics = BinaryMetricComputer({"Anthropogenic": 0, "Biogenic": 1})
    X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=2, random_state=0)
    inner_clf = LogisticRegression(solver="liblinear", random_state=0)
    clf = MultiOutputClassifier(inner_clf).fit(X, y)
    probs = clf.predict_proba(X)

    y_score = np.transpose([y_pred[:, 1] for y_pred in probs]) # the by-label probabilities per sample. E.g. y_score[14, 3] tells us the probability of sample 14 to have a positive instance of label/class 3
    truth = torch.tensor(y, dtype=torch.float32, requires_grad=False)
    preds = torch.tensor(y_score, dtype=torch.float32, requires_grad=False)
    print(metrics(truth, preds))
    