"""End-to-end pipeline: waveforms -> WST -> classifier -> metrics.

This is the "just run it" entry point. Assuming the dataset has already been
downloaded (see :mod:`wst_eccentricity.data`) into a folder laid out as::

    data_dir/
        waveforms/     # <name>.hdf5 per signal
        parameters/    # params_*.txt per signal

a single call computes the wavelet scattering transform (caching it under
``data_dir/transform_coefficients/``), builds a labelled dataset, trains the
chosen classifier, and reports AUC, average precision and the true-positive rate
at a target false-positive rate.

Command line::

    # 1. download the example dataset into ./data
    python -m wst_eccentricity.data --dest data

    # 2. run the WST + a classifier of your choice
    wst-eccentricity-run --data-dir data --classifier cnn1d --J 7 --Q 2
    #  (equivalently: python -m wst_eccentricity.pipeline --data-dir data ...)

Available classifiers are listed in :data:`CLASSIFIERS`. To add your own, insert
a function into that dict (see :func:`register_classifier`).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .datasets import build_dataset, flatten_features, standardize
from .io import load_hdf5_data
from .metrics import (
    auc_ap,
    confusion_counts,
    fpr_tpr_from_counts,
    threshold_for_target_fpr,
)
from .transforms import scatter_in_batches

__all__ = ["run_pipeline", "CLASSIFIERS", "register_classifier"]


# --------------------------------------------------------------------------- #
# Classifier registry.  Each entry maps a name to a callable with the signature
#     fn(X_train, y_train, X_val, y_val, X_test, device, **kwargs)
#         -> (val_probs, test_probs)
# where the returned arrays are positive-class probabilities.
# --------------------------------------------------------------------------- #

CLASSIFIERS: dict = {}


def register_classifier(name: str):
    """Decorator that registers a classifier under ``name`` in :data:`CLASSIFIERS`.

    A classifier is any callable
    ``fn(X_train, y_train, X_val, y_val, X_test, device, **kwargs)`` returning
    ``(val_probs, test_probs)`` -- the positive-class probabilities for the
    validation and test sets.
    """

    def wrap(fn):
        CLASSIFIERS[name] = fn
        return fn

    return wrap


@register_classifier("cnn1d")
def _train_cnn1d(
    X_train, y_train, X_val, y_val, X_test, device,
    epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, patience: int = 5,
    **model_kwargs,
):
    """The paper's reference 1D-CNN (:class:`~wst_eccentricity.models.Conv1DNet`)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .metrics import collect_probs_targets
    from .models import Conv1DNet
    from .training import train_binary

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
    test_loader = DataLoader(
        TensorDataset(X_test, torch.zeros(len(X_test), dtype=torch.long)), batch_size=256
    )

    allowed = {"in_channels", "num_filters", "kernel_size", "num_conv_layers", "dropout"}
    model = Conv1DNet(
        input_size=X_train.shape[1],
        **{k: v for k, v in model_kwargs.items() if k in allowed},
    )
    model, _history, _auc = train_binary(
        model, train_loader, val_loader, lr=lr, max_epochs=epochs,
        patience=patience, device=device,
    )
    _, val_probs = collect_probs_targets(model, val_loader, device)
    _, test_probs = collect_probs_targets(model, test_loader, device)
    return val_probs, test_probs


@register_classifier("logreg")
def _train_logreg(X_train, y_train, X_val, y_val, X_test, device, max_iter: int = 1000, **_):
    """A simple logistic-regression baseline (scikit-learn)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train.numpy(), y_train.numpy())
    val_probs = clf.predict_proba(X_val.numpy())[:, 1]
    test_probs = clf.predict_proba(X_test.numpy())[:, 1]
    return val_probs, test_probs


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _stratified_split(y, val_frac: float, test_frac: float, seed: int):
    """Return ``(train_idx, val_idx, test_idx)`` with class-stratified sampling."""
    from sklearn.model_selection import train_test_split

    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    idx = np.arange(len(y_np))
    train_idx, tmp_idx = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=seed, stratify=y_np
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=rel_test, random_state=seed, stratify=y_np[tmp_idx]
    )
    return train_idx, val_idx, test_idx


def run_pipeline(
    data_dir: str,
    J: int = 7,
    Q: int = 2,
    classifier: str = "cnn1d",
    e_thr: float = 0.01,
    target_fpr: float = 0.1,
    recompute_wst: bool = False,
    device: str | None = None,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    **classifier_kwargs,
) -> dict:
    """Run the full WST + classification pipeline on a prepared dataset.

    Parameters
    ----------
    data_dir:
        Folder containing ``waveforms/`` and ``parameters/`` (as downloaded from
        Zenodo). WST coefficients are cached in ``data_dir/transform_coefficients``.
    J, Q:
        Wavelet scattering hyper-parameters.
    classifier:
        Name of the classifier to train; must be a key of :data:`CLASSIFIERS`
        (built-in: ``"cnn1d"``, ``"logreg"``).
    e_thr:
        Eccentricity threshold for the binary label (``eccentricity > e_thr``).
    target_fpr:
        False-positive rate at which the operating threshold is set (on the
        validation split) before evaluating on the test split.
    recompute_wst:
        Force recomputation of the WST even if cached files exist.
    device:
        Compute device; defaults to CUDA if available, else CPU.
    val_frac, test_frac:
        Fractions of the data used for validation and test (the rest is training).
    seed:
        Random seed for the stratified split.
    **classifier_kwargs:
        Extra keyword arguments forwarded to the chosen classifier (e.g.
        ``epochs``, ``num_filters`` for ``cnn1d``).

    Returns
    -------
    dict
        Metrics: ``auc``, ``average_precision``, ``tau`` (threshold),
        ``fpr``, ``tpr``, plus the settings used.
    """
    if classifier not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier {classifier!r}. Available: {sorted(CLASSIFIERS)}"
        )

    device = _resolve_device(device)
    waveforms_dir = os.path.join(data_dir, "waveforms")
    params_dir = os.path.join(data_dir, "parameters")
    wst_dir = os.path.join(data_dir, "transform_coefficients")

    # 1. Wavelet scattering transform (cached).
    cached = glob.glob(os.path.join(wst_dir, f"WST_{J}_{Q}_*.pt"))
    if recompute_wst or not cached:
        print(f"[1/3] Computing WST (J={J}, Q={Q}) on {device} ...")
        gws, _ = load_hdf5_data(waveforms_dir)
        scatter_in_batches(gws, J, Q, out_dir=wst_dir, device=device)
    else:
        print(f"[1/3] Using cached WST in {wst_dir}")

    # 2. Build labelled dataset and split.
    print("[2/3] Building labelled dataset ...")
    Sx, y, _params = build_dataset(params_dir, wst_dir, J, Q, e_thr=e_thr)
    X = standardize(flatten_features(Sx))
    tr, va, te = _stratified_split(y, val_frac, test_frac, seed)

    import torch

    tr, va, te = (torch.as_tensor(i) for i in (tr, va, te))

    # 3. Train the chosen classifier and evaluate.
    print(f"[3/3] Training classifier {classifier!r} ...")
    val_probs, test_probs = CLASSIFIERS[classifier](
        X[tr], y[tr], X[va], y[va], X[te], device, **classifier_kwargs
    )

    from sklearn.metrics import roc_curve

    y_val = y[va].numpy()
    y_test = y[te].numpy()
    fpr, tpr, thr = roc_curve(y_val, val_probs, pos_label=1)
    tau, _, _ = threshold_for_target_fpr(fpr, tpr, thr, target_fpr)

    auc, ap = auc_ap(y_test, test_probs)
    tn, fp, fn, tp = confusion_counts(y_test, test_probs, tau)
    fpr_at, tpr_at = fpr_tpr_from_counts(tn, fp, fn, tp)

    results = {
        "classifier": classifier,
        "J": J,
        "Q": Q,
        "n_signals": int(len(y)),
        "auc": auc,
        "average_precision": ap,
        "target_fpr": target_fpr,
        "tau": tau,
        "fpr": fpr_at,
        "tpr": tpr_at,
    }

    print("\n=== Results ===")
    print(f"classifier:          {classifier}  (J={J}, Q={Q}, N={len(y)})")
    print(f"AUC:                 {auc:.3f}")
    print(f"average precision:   {ap:.3f}")
    print(f"at target FPR={target_fpr:.2f}: FPR={fpr_at:.3f}, TPR={tpr_at:.3f} (tau={tau:.3f})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WST + classifier on a prepared dataset.")
    parser.add_argument("--data-dir", required=True, help="folder with waveforms/ and parameters/")
    parser.add_argument("--classifier", default="cnn1d", choices=sorted(CLASSIFIERS),
                        help="classifier to train (default: cnn1d)")
    parser.add_argument("--J", type=int, default=7, help="max scattering scale")
    parser.add_argument("--Q", type=int, default=2, help="wavelets per octave")
    parser.add_argument("--e-thr", type=float, default=0.01, help="eccentricity label threshold")
    parser.add_argument("--target-fpr", type=float, default=0.1, help="operating false-positive rate")
    parser.add_argument("--epochs", type=int, default=50, help="epochs (cnn1d only)")
    parser.add_argument("--recompute-wst", action="store_true", help="ignore cached WST files")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        J=args.J,
        Q=args.Q,
        classifier=args.classifier,
        e_thr=args.e_thr,
        target_fpr=args.target_fpr,
        recompute_wst=args.recompute_wst,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
