"""End-to-end quickstart for the wst-eccentricity pipeline.

Two modes:

* ``--demo``: run on small random tensors so you can exercise the training /
  evaluation API without any data (handy for a smoke test).
* ``--data-dir PATH``: run on a real, pre-generated dataset laid out as::

      PATH/
          waveforms/    # <name>.hdf5 per signal
          parameters/   # params_*.txt per signal

  The script computes the WST, builds a labelled dataset, trains a
  :class:`~wst_eccentricity.models.Conv1DNet`, and reports AUC and the true
  positive rate at a 10% false positive rate.

Examples
--------
    python examples/quickstart.py --demo
    python examples/quickstart.py --download            # fetch the Zenodo example set
    python examples/quickstart.py --data-dir /path/to/dataset --J 7 --Q 2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, TensorDataset, random_split

from wst_eccentricity import (
    Conv1DNet,
    auc_ap,
    build_dataset,
    collect_probs_targets,
    compute_scattering,
    flatten_features,
    fpr_tpr_from_counts,
    standardize,
    threshold_for_target_fpr,
)
from wst_eccentricity.metrics import confusion_counts


def load_features(args):
    """Return ``(features, labels)`` either from a real dataset or synthetically."""
    if args.demo:
        # Synthetic: two Gaussian blobs so the classifier has something to learn.
        torch.manual_seed(0)
        n, feat = 400, 256
        y = torch.randint(0, 2, (n,))
        X = torch.randn(n, feat) + y.unsqueeze(1) * 0.5
        return X, y.long()

    # Real data: waveforms -> WST -> labelled dataset.
    from wst_eccentricity import load_hdf5_data

    gws, _ = load_hdf5_data(f"{args.data_dir}/waveforms")
    scatter_dir = f"{args.data_dir}/transform_coefficients"
    from wst_eccentricity import scatter_in_batches

    scatter_in_batches(gws, args.J, args.Q, out_dir=scatter_dir, device=args.device)
    Sx, y, _params = build_dataset(
        params_dir=f"{args.data_dir}/parameters",
        wst_dir=scatter_dir,
        J=args.J,
        Q=args.Q,
        e_thr=args.e_thr,
    )
    return flatten_features(Sx), y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run on synthetic data")
    parser.add_argument("--download", action="store_true",
                        help="download the Zenodo example dataset and run on it")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--Q", type=int, default=2)
    parser.add_argument("--e-thr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.download and args.data_dir is None:
        from wst_eccentricity import download_example_data

        args.data_dir = str(download_example_data("data"))

    if not args.demo and args.data_dir is None:
        parser.error(
            "provide --data-dir PATH, use --download to fetch the example set, "
            "or --demo for a synthetic run"
        )

    X, y = load_features(args)
    X = standardize(X)

    ds = TensorDataset(X, y)
    n_val = max(1, int(0.2 * len(ds)))
    n_test = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    from wst_eccentricity import train_binary

    model = Conv1DNet(input_size=X.shape[1])
    model, _history, best_val_auc = train_binary(
        model, train_loader, val_loader,
        max_epochs=args.epochs, device=args.device,
    )
    print(f"Best validation AUC: {best_val_auc:.3f}")

    # Choose the operating threshold on validation, evaluate on test.
    val_t, val_p = collect_probs_targets(model, val_loader, args.device)
    fpr, tpr, thr = roc_curve(val_t, val_p, pos_label=1)
    tau, _, _ = threshold_for_target_fpr(fpr, tpr, thr, args.target_fpr)

    test_t, test_p = collect_probs_targets(model, test_loader, args.device)
    test_auc, test_ap = auc_ap(test_t, test_p)
    tn, fp, fn, tp = confusion_counts(test_t, test_p, tau)
    fpr_at, tpr_at = fpr_tpr_from_counts(tn, fp, fn, tp)

    print(f"Test AUC:               {test_auc:.3f}")
    print(f"Test average precision: {test_ap:.3f}")
    print(f"At target FPR={args.target_fpr:.2f} (tau={tau:.3f}): "
          f"FPR={fpr_at:.3f}, TPR={tpr_at:.3f}")


if __name__ == "__main__":
    main()
