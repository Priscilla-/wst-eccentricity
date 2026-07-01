"""Input/output helpers for waveforms, parameters and scattering coefficients.

This module assumes that the gravitational-wave dataset has *already* been
generated (waveform generation is intentionally out of scope for this package).
It provides small, dependency-light utilities to:

* load a folder of per-signal waveform ``.hdf5`` files into a single array,
* read the plain-text parameter files that accompany each signal, and
* load/save wavelet-scattering-transform (WST) coefficient tensors (``.pt``).

The expected on-disk layout is::

    data_dir/
        waveforms/       # one <name>.hdf5 per signal, each holding one array
        parameters/      # one params_*.txt per signal (``key: value`` lines)
        transform_coefficients/   # WST_{J}_{Q}_*.pt written by :mod:`~wst_eccentricity.transforms`
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

__all__ = [
    "get_seed_and_index",
    "load_hdf5_data",
    "read_parameters",
    "load_scattering_data",
]


def get_seed_and_index(path: str) -> tuple[int, int]:
    """Extract the ``(seed, index)`` pair encoded in a file name.

    File names produced by the data-generation pipeline contain a tag of the
    form ``s<seed>_<index>`` (for example ``params_s30_10002.txt``). Sorting by
    this pair guarantees that waveforms and their parameter files stay aligned.

    Parameters
    ----------
    path:
        File path or name to parse.

    Returns
    -------
    tuple of int
        ``(seed, index)`` if the pattern is found, otherwise a large sentinel
        ``(9999, 999999)`` so that unmatched files sort last.
    """
    match = re.search(r"s(\d+)_(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1)), int(match.group(2))
    return (9999, 999999)


def load_hdf5_data(
    folder: str,
    np_dtype: type = np.float32,
    sort_by_seed: bool = True,
    start_from: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """Load every ``.hdf5`` waveform file in ``folder`` into one tensor.

    Each file is assumed to contain a single dataset, all of the same shape
    (for example ``(n_detectors, n_samples)``).

    Parameters
    ----------
    folder:
        Directory containing the ``.hdf5`` files.
    np_dtype:
        NumPy dtype used for the pre-allocated output array.
    sort_by_seed:
        If ``True`` (default), sort files by their ``(seed, index)`` tag so the
        ordering matches :func:`read_parameters`. If ``False``, sort
        alphabetically.
    start_from:
        Skip the first ``start_from`` files (useful for resuming).

    Returns
    -------
    data : torch.Tensor
        Tensor of shape ``(n_files, *signal_shape)``.
    files : list of str
        The file paths, in the same order as ``data``.
    """
    if sort_by_seed:
        files = sorted(glob.glob(os.path.join(folder, "*.hdf5")), key=get_seed_and_index)
    else:
        files = sorted(glob.glob(os.path.join(folder, "*.hdf5")))

    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {folder!r}")

    import h5py  # local import so the package imports without h5py present

    # Infer the dataset key and shape from the first file.
    with h5py.File(files[0], "r") as file0:
        dataset_key = next(
            (name for name in file0 if isinstance(file0[name], h5py.Dataset)), None
        )
        if dataset_key is None:
            raise ValueError(f"No dataset found inside {files[0]!r}")
        shape = file0[dataset_key].shape

    selected = files[start_from:]
    out = np.empty((len(selected),) + shape, dtype=np_dtype)
    for i, path in enumerate(selected):
        with h5py.File(path, "r") as f:
            data = f[dataset_key]
            if data.shape != shape:
                raise ValueError(
                    f"Shape mismatch in {os.path.basename(path)}: "
                    f"{data.shape} != {shape}"
                )
            data.read_direct(out[i])

    return torch.from_numpy(out), selected


def read_parameters(
    folder: str,
    pattern: str = "*.txt",
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Read the plain-text parameter files in ``folder``.

    Each file is expected to contain ``key: value`` lines, e.g.::

        eccentricity: 0.023
        mass_1: 35.0
        NSNR: 42

    Parameters
    ----------
    folder:
        Directory containing the parameter files.
    pattern:
        Glob pattern selecting the files (default ``"*.txt"``).
    verbose:
        If ``True``, print the first few file names for a sanity check.

    Returns
    -------
    list of dict
        One ``{parameter_name: float_value}`` dictionary per file, ordered by
        the ``(seed, index)`` tag so the ordering matches :func:`load_hdf5_data`.
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=get_seed_and_index)
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in: {folder!r}")
    if verbose:
        print("First parameter files:", files[:4])

    records: list[dict[str, float]] = []
    for path in files:
        record: dict[str, float] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                record[key.strip()] = float(val.strip())
        records.append(record)
    return records


def load_scattering_data(path: str, device: str = "cpu"):
    """Load a scattering-coefficient tensor saved by this package.

    Parameters
    ----------
    path:
        Path to a ``.pt`` file written by
        :func:`wst_eccentricity.transforms.scatter_in_batches`.
    device:
        Device to move the tensor onto (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    Sx : torch.Tensor
        The scattering coefficients, shape ``(N, n_detectors, C, T)``.
    config : dict
        The configuration used to compute them (``J``, ``Q``, ...).
    meta : dict or None
        Kymatio metadata describing each coefficient (order, path, ...).
    """
    data = torch.load(path, map_location=device, weights_only=False)
    return data["Sx"].to(device), data["config"], data.get("meta", None)
