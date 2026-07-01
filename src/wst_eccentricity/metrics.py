"""Evaluation metrics for the eccentricity classifier.

Following the paper, performance is reported at fixed operating points along
the **false positive rate** (FPR) axis -- the fraction of quasi-circular
signals wrongly labelled eccentric -- together with the true positive rate
(TPR), the area under the ROC curve (AUC) and the average precision (AP).

The FPR is dimensionless (a probability on the test set), not a rate per unit
time.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "confusion_counts",
    "fpr_tpr_from_counts",
    "threshold_for_target_fpr",
    "auc_ap",
    "collect_probs_targets",
]


def confusion_counts(
    targets: np.ndarray, probs: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for a given decision threshold.

    Parameters
    ----------
    targets:
        True binary labels.
    probs:
        Predicted probabilities for the positive class.
    threshold:
        Probabilities ``>= threshold`` are predicted positive.

    Returns
    -------
    tuple of int
        True negatives, false positives, false negatives, true positives.
    """
    from sklearn.metrics import confusion_matrix

    targets = np.asarray(targets).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    preds = (probs >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def fpr_tpr_from_counts(tn: int, fp: int, fn: int, tp: int) -> tuple[float, float]:
    """Compute ``(fpr, tpr)`` from confusion-matrix counts."""
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return float(fpr), float(tpr)


def threshold_for_target_fpr(
    fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray, target_fpr: float
) -> tuple[float, float, float]:
    """Pick the decision threshold that best matches a target FPR.

    Given the arrays returned by :func:`sklearn.metrics.roc_curve`, choose the
    threshold whose FPR is closest to ``target_fpr``, preferring a conservative
    operating point (``fpr <= target_fpr`` and the larger threshold when tied).

    Parameters
    ----------
    fpr, tpr, thr:
        Outputs of :func:`sklearn.metrics.roc_curve`.
    target_fpr:
        Desired false positive rate (e.g. ``0.1`` for 10%).

    Returns
    -------
    tuple of float
        ``(threshold, achieved_fpr, achieved_tpr)``.
    """
    target_fpr = float(target_fpr)
    diffs = np.abs(fpr - target_fpr)
    cand = np.where(diffs == diffs.min())[0]

    le = cand[fpr[cand] <= target_fpr]
    if len(le) > 0:
        cand = le
    i = cand[np.argmax(thr[cand])]
    return float(thr[i]), float(fpr[i]), float(tpr[i])


def auc_ap(targets: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Return ``(AUC, average_precision)``; ``(nan, nan)`` if only one class present."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    targets = np.asarray(targets).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    if len(np.unique(targets)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(targets, probs)), float(average_precision_score(targets, probs))


@torch.no_grad()
def collect_probs_targets(model, loader, device):
    """Run ``model`` over ``loader`` and collect predicted probabilities and labels.

    Parameters
    ----------
    model:
        A trained model returning one logit per sample.
    loader:
        A data loader yielding ``(x, y)`` batches.
    device:
        Device to run inference on.

    Returns
    -------
    targets : numpy.ndarray
        True labels.
    probs : numpy.ndarray
        Predicted positive-class probabilities.
    """
    model.eval()
    all_targets, all_probs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        all_targets.append(yb.detach().cpu().numpy().astype(float).ravel())
        all_probs.append(probs.detach().cpu().numpy().ravel())
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    return targets, probs
