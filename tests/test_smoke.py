"""Fast smoke tests that require only numpy, torch and scikit-learn.

These check that the core API is wired together correctly; they do not need any
data, Kymatio, or h5py.
"""

import numpy as np
import torch

from wst_eccentricity import (
    Conv1DNet,
    auc_ap,
    class_from_eccentricity,
    confusion_counts,
    flatten_features,
    fpr_tpr_from_counts,
    standardize,
    threshold_for_target_fpr,
    train_binary,
)


def test_class_from_eccentricity():
    ecc = torch.tensor([0.0, 0.005, 0.01, 0.02, 0.5])
    labels = class_from_eccentricity(ecc, e_thr=0.01)
    assert labels.tolist() == [0, 0, 0, 1, 1]
    assert labels.dtype == torch.long


def test_flatten_and_standardize():
    Sx = torch.randn(8, 3, 4, 5)  # (N, D, C, T)
    X = flatten_features(Sx)
    assert X.shape == (8, 3 * 4 * 5)
    Xs = standardize(X)
    assert torch.isfinite(Xs).all()


def test_conv1dnet_forward():
    model = Conv1DNet(input_size=64)
    out = model(torch.randn(5, 64))
    assert out.shape == (5,)


def test_metrics_and_threshold():
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([rng.uniform(0, 0.6, 50), rng.uniform(0.4, 1.0, 50)])
    auc, ap = auc_ap(y, p)
    assert 0.0 <= auc <= 1.0 and 0.0 <= ap <= 1.0

    fpr, tpr, thr = roc_curve(y, p, pos_label=1)
    tau, got_fpr, got_tpr = threshold_for_target_fpr(fpr, tpr, thr, 0.1)
    tn, fp, fn, tp = confusion_counts(y, p, tau)
    f, t = fpr_tpr_from_counts(tn, fp, fn, tp)
    assert 0.0 <= f <= 1.0 and 0.0 <= t <= 1.0


def test_train_binary_runs():
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    y = torch.randint(0, 2, (120,))
    X = torch.randn(120, 32) + y.unsqueeze(1) * 0.6
    ds = TensorDataset(standardize(X), y.long())
    train = DataLoader(ds, batch_size=32, shuffle=True)
    val = DataLoader(ds, batch_size=64)

    model = Conv1DNet(input_size=32, num_filters=8)
    model, history, best_auc = train_binary(model, train, val, max_epochs=3, patience=5)
    assert len(history["train_loss"]) >= 1
    assert np.isfinite(best_auc)
