#!/usr/bin/env python3
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=2, random_state=0)
print(X)
print(X.shape)

print(y[:10])
print(y.shape)
inner_clf = LogisticRegression(solver="liblinear", random_state=0)
clf = MultiOutputClassifier(inner_clf).fit(X, y)
probs = clf.predict_proba(X)
print(len(probs))
for clsidx in range(len(probs)):
    print(len(probs[clsidx]))
    for sampleidx in range(len(probs[clsidx])):
        prob_false, prob_true = probs[clsidx][sampleidx]
        print(prob_false, prob_true, prob_true+prob_false)


y_score = np.transpose([y_pred[:, 1] for y_pred in probs]) # the by-label probabilities per sample. E.g. y_score[14, 3] tells us the probability of sample 14 to have a positive instance of label/class 3
print(y_score.shape)
for row in y_score:
    print(np.sum(row))
print(y)
print(y_score[:10])
print(y_score.shape)
roc = roc_auc_score(y, y_score, average=None)
print(roc)