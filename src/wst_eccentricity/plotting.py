"""Lightweight plotting helpers (optional, require :mod:`matplotlib`).

These are convenience functions for inspecting a trained classifier: the ROC
curve, the confusion matrix, and the training/validation loss history.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_roc_curve", "plot_confusion_matrix", "plot_history"]


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    target_fprs: tuple[float, ...] = (0.1, 0.01),
    log_scale: bool = True,
    ax=None,
):
    """Plot an ROC curve with vertical markers at target FPRs.

    Parameters
    ----------
    fpr, tpr:
        False- and true-positive-rate arrays (from ``roc_curve``).
    auc:
        AUC value to annotate.
    target_fprs:
        FPR values at which to draw dashed vertical lines.
    log_scale:
        Use a logarithmic x-axis.
    ax:
        Optional existing matplotlib axis to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlim(min(1e-3, min(target_fprs) / 2), 1.0)
    for f in target_fprs:
        ax.axvline(f, color="black", linestyle="dashed", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    return ax


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, normalize: bool = True, ax=None):
    """Plot a 2x2 confusion matrix.

    Parameters
    ----------
    tn, fp, fn, tp:
        Confusion-matrix counts.
    normalize:
        If ``True``, annotate each cell with the row-normalised fraction as well
        as the raw count.
    ax:
        Optional existing matplotlib axis to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    import matplotlib.pyplot as plt

    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    disp = cm / cm.sum(axis=1, keepdims=True) if normalize else cm

    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(disp, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    thr = disp.max() / 2.0 if disp.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            txt = f"{int(cm[i, j])}" + (f"\n{disp[i, j] * 100:.1f}%" if normalize else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thr else "black")
    im.axes.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_history(history: dict, ax=None):
    """Plot training and validation loss versus epoch.

    Parameters
    ----------
    history:
        The ``history`` dict returned by
        :func:`wst_eccentricity.training.train_binary`.
    ax:
        Optional existing matplotlib axis to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    if history.get("val_auc"):
        best = int(np.argmax(history["val_auc"])) + 1
        ax.axvline(best, color="red", linestyle="--", label=f"Best AUC (epoch {best})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(color="gainsboro", alpha=0.7)
    return ax
