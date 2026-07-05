"""Fast smoke tests that require only numpy, torch and scikit-learn.

These check that the core API is wired together correctly; they do not need any
data, Kymatio, or h5py.
"""

import numpy as np
import torch

from wst_eccentricity import (
    SWT_CNN_1D_Binned,
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


def test_swt_cnn_1d_binned_forward():
    # (batch, detectors, channels, time)
    model = SWT_CNN_1D_Binned(in_channels=4, num_detectors=3, time_bins=4)
    out = model(torch.randn(5, 3, 4, 16))
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


def test_match_waveform_and_parameter_files(tmp_path):
    import pytest

    from wst_eccentricity import match_waveform_and_parameter_files, read_parameters

    wdir = tmp_path / "waveforms"
    pdir = tmp_path / "parameters"
    wdir.mkdir()
    pdir.mkdir()
    for i in (0, 2, 10):
        (wdir / f"waveform_s42_{i}_0.1_30.0_20.0_50.hdf5").touch()
        (pdir / f"params_s42_{i}.txt").write_text(f"eccentricity: 0.{i}\nNSNR: 50\n")

    wf, pf = match_waveform_and_parameter_files(str(wdir), str(pdir))
    assert len(wf) == len(pf) == 3
    # Sorted numerically by (seed, index), not alphabetically.
    assert [p.split("_")[-1] for p in pf] == ["0.txt", "2.txt", "10.txt"]

    records = read_parameters(files=pf)
    assert [r["eccentricity"] for r in records] == [0.0, 0.2, 0.10]

    # A parameter file without a waveform must raise in strict mode ...
    (pdir / "params_s42_99.txt").write_text("eccentricity: 0.3\n")
    with pytest.raises(ValueError):
        match_waveform_and_parameter_files(str(wdir), str(pdir))
    # ... and be dropped when strict=False.
    wf, pf = match_waveform_and_parameter_files(str(wdir), str(pdir), strict=False)
    assert len(wf) == len(pf) == 3


def test_read_parameters_skips_non_numeric(tmp_path):
    from wst_eccentricity import read_parameters

    f = tmp_path / "params_s1_0.txt"
    f.write_text("eccentricity: 0.05\napproximant: SEOBNRv5EHM\nNSNR: 20\n")
    (record,) = read_parameters(files=[str(f)])
    assert record == {"eccentricity": 0.05, "NSNR": 20.0}


def test_train_binary_runs():
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    D, C, T = 3, 4, 16
    y = torch.randint(0, 2, (120,))
    # (N, D, C, T) with a mild class-dependent shift so there is signal to learn.
    X = torch.randn(120, D, C, T) + y.view(-1, 1, 1, 1) * 0.6
    ds = TensorDataset(standardize(X), y.long())
    train = DataLoader(ds, batch_size=32, shuffle=True)
    val = DataLoader(ds, batch_size=64)

    model = SWT_CNN_1D_Binned(in_channels=C, num_detectors=D, time_bins=4)
    model, history, best_auc = train_binary(model, train, val, max_epochs=3, patience=5)
    assert len(history["train_loss"]) >= 1
    assert np.isfinite(best_auc)
