from sklearn import metrics
from typing import Any
import numpy as np


class Metric(object):
    def __init__(self, cfgs) -> None:
        pass

    def __call__(self, target, pred):
        """
        target: [n_sample, 1]
        pred: [n_sample, n_classes]
        """
        pred = np.array(pred).reshape([-1, 2])
        probs = pred[:, 1]
        pred = np.argmax(pred, axis=1)
        tn, fp, fn, tp = metrics.confusion_matrix(target, pred).ravel()
        score = {
            "Pre.": metrics.precision_score(target, pred),
            "Rec.": metrics.recall_score(target, pred),
            "F1": metrics.f1_score(target, pred),
            "Acc.": metrics.accuracy_score(target, pred),
            "AUC": metrics.roc_auc_score(target, probs),
            "Spe.": tn / (tn+fp)
        }
        return score
