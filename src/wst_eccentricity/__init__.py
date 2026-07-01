"""wst-eccentricity: rapid classification of eccentric binaries with the WST.

A compact, reproducible pipeline that takes *pre-generated* gravitational-wave
strain data, computes its wavelet scattering transform (WST), and trains a
lightweight 1D convolutional neural network to distinguish eccentric from
quasi-circular compact-binary signals.

Waveform generation is intentionally out of scope: the pipeline starts from a
provided waveform dataset and its parameter files.

Typical usage::

    from wst_eccentricity import (
        compute_scattering, build_dataset, SWT_CNN_1D_Binned, train_binary,
    )

See the ``examples/`` directory for an end-to-end script.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data import download_example_data
from .datasets import (
    H5ScatteringDataset,
    build_dataset,
    class_from_eccentricity,
    flatten_features,
    labels_from_params,
    load_scattering_batches,
    load_scattering_hdf5,
    save_scattering_hdf5,
    standardize,
)
from .io import load_hdf5_data, load_scattering_data, read_parameters
from .metrics import (
    auc_ap,
    collect_probs_targets,
    confusion_counts,
    fpr_tpr_from_counts,
    threshold_for_target_fpr,
)
from .models import SWT_CNN_1D_Binned
from .pipeline import CLASSIFIERS, register_classifier, run_pipeline
from .training import train_binary
from .transforms import compute_scattering, scatter_in_batches

__all__ = [
    "__version__",
    # example data
    "download_example_data",
    # transforms
    "compute_scattering",
    "scatter_in_batches",
    # io
    "load_hdf5_data",
    "read_parameters",
    "load_scattering_data",
    # datasets
    "build_dataset",
    "load_scattering_batches",
    "class_from_eccentricity",
    "labels_from_params",
    "flatten_features",
    "standardize",
    "save_scattering_hdf5",
    "load_scattering_hdf5",
    "H5ScatteringDataset",
    # models / training
    "SWT_CNN_1D_Binned",
    "train_binary",
    # end-to-end pipeline
    "run_pipeline",
    "CLASSIFIERS",
    "register_classifier",
    # metrics
    "confusion_counts",
    "fpr_tpr_from_counts",
    "threshold_for_target_fpr",
    "auc_ap",
    "collect_probs_targets",
]
