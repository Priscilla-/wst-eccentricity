"""Build labelled datasets from scattering coefficients and parameter files.

The public pipeline assumes the waveform dataset and its parameter files are
provided. Given a folder of WST ``.pt`` batch files (see
:mod:`wst_eccentricity.transforms`) and the matching parameter folder, the
helpers here assemble a single ``(features, labels)`` dataset ready for the
classifier.

Eccentricity is turned into a binary label with a single threshold
``e_thr`` (the paper uses ``e_thr = 0.01``): signals with
``eccentricity <= e_thr`` are the negative ("quasi-circular") class and the
rest are the positive ("eccentric") class.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_scattering_data, read_parameters

__all__ = [
    "standardize",
    "class_from_eccentricity",
    "labels_from_params",
    "load_scattering_batches",
    "build_dataset",
    "flatten_features",
    "save_scattering_hdf5",
    "load_scattering_hdf5",
    "H5ScatteringDataset",
]


def standardize(data: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    """Standardize features to zero mean and unit variance along the batch axis.

    Parameters
    ----------
    data:
        Tensor whose first dimension indexes samples.
    eps:
        Small constant added to the standard deviation to avoid division by
        zero.

    Returns
    -------
    torch.Tensor
        The standardized tensor, same shape as ``data``.
    """
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / (std + eps)


def class_from_eccentricity(
    eccentricities: torch.Tensor, e_thr: float = 0.01
) -> torch.Tensor:
    """Map eccentricity values to binary class labels.

    Parameters
    ----------
    eccentricities:
        1D tensor of eccentricity values.
    e_thr:
        Decision threshold. ``eccentricity <= e_thr`` -> class ``0``
        (quasi-circular); ``eccentricity > e_thr`` -> class ``1`` (eccentric).

    Returns
    -------
    torch.Tensor
        Integer (``long``) tensor of 0/1 labels, same length as the input.
    """
    labels = torch.where(
        eccentricities > e_thr,
        torch.ones_like(eccentricities),
        torch.zeros_like(eccentricities),
    ).long()
    return labels


def labels_from_params(
    params_dir: str, param_keys: tuple[str, ...] = ("eccentricity",)
) -> torch.Tensor:
    """Read selected parameters from a folder into a ``(N, len(param_keys))`` tensor.

    Parameters
    ----------
    params_dir:
        Folder containing the ``params_*.txt`` files.
    param_keys:
        Which parameters to extract, in column order.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(n_signals, len(param_keys))``.
    """
    records = read_parameters(params_dir)
    out = torch.zeros(len(records), len(param_keys))
    for i, record in enumerate(records):
        for j, key in enumerate(param_keys):
            out[i, j] = record[key]
    return out


def _sort_batch_files(files: list[str]) -> list[str]:
    """Sort ``WST_{J}_{Q}_{batch}.pt`` files by their numeric batch index."""

    def key(path: str):
        base = os.path.basename(path)
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", base)]

    return sorted(files, key=key)


def load_scattering_batches(
    wst_dir: str, J: int, Q: int, n_expected: int | None = None
) -> torch.Tensor:
    """Load and concatenate all WST batch files for a given ``(J, Q)``.

    Parameters
    ----------
    wst_dir:
        Directory holding ``WST_{J}_{Q}_*.pt`` files.
    J, Q:
        Scattering hyper-parameters selecting which files to read.
    n_expected:
        Optional expected number of signals; if given, a mismatch raises an
        error (a useful guard against missing batches).

    Returns
    -------
    torch.Tensor
        Coefficients of shape ``(N, D, C, T)``.
    """
    files = _sort_batch_files(glob.glob(os.path.join(wst_dir, f"WST_{J}_{Q}_*.pt")))
    if not files:
        raise FileNotFoundError(
            f"No WST_{J}_{Q}_*.pt files found in {wst_dir!r}"
        )
    chunks = [load_scattering_data(f)[0] for f in files]
    Sx = torch.cat(chunks, dim=0)
    if n_expected is not None and Sx.shape[0] != n_expected:
        raise ValueError(
            f"Loaded {Sx.shape[0]} signals but expected {n_expected}."
        )
    return Sx


def build_dataset(
    params_dir: str,
    wst_dir: str,
    J: int,
    Q: int,
    e_thr: float = 0.01,
    param_keys: tuple[str, ...] = ("eccentricity",),
):
    """Assemble a labelled dataset for a single ``(J, Q)`` scattering setting.

    Parameters
    ----------
    params_dir:
        Folder with the ``params_*.txt`` parameter files.
    wst_dir:
        Folder with the ``WST_{J}_{Q}_*.pt`` scattering batch files.
    J, Q:
        Scattering hyper-parameters.
    e_thr:
        Eccentricity threshold used to build binary labels.
    param_keys:
        Physical parameters to return alongside the features (the first key
        must be ``"eccentricity"`` so labels can be derived from it).

    Returns
    -------
    Sx : torch.Tensor
        Scattering features of shape ``(N, D, C, T)``.
    y : torch.Tensor
        Binary labels of shape ``(N,)``.
    params : torch.Tensor
        Physical parameters of shape ``(N, len(param_keys))``.
    """
    if param_keys[0] != "eccentricity":
        raise ValueError("The first entry of param_keys must be 'eccentricity'.")

    params = labels_from_params(params_dir, param_keys=param_keys)
    n = params.shape[0]
    Sx = load_scattering_batches(wst_dir, J, Q, n_expected=n)
    y = class_from_eccentricity(params[:, 0], e_thr=e_thr)
    return Sx, y, params


def flatten_features(Sx: torch.Tensor) -> torch.Tensor:
    """Flatten scattering features to ``(N, D * C * T)`` for a 1D-CNN.

    Parameters
    ----------
    Sx:
        Tensor of shape ``(N, D, C, T)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, D * C * T)``.
    """
    return Sx.reshape(Sx.shape[0], -1)


# --------------------------------------------------------------------------- #
# Optional HDF5 cache: build once, reload fast.
# --------------------------------------------------------------------------- #

def save_scattering_hdf5(
    h5_path: str,
    y: torch.Tensor,
    Sx: torch.Tensor,
    params: torch.Tensor | None = None,
) -> None:
    """Cache features and labels to an HDF5 file (order-preserving).

    Parameters
    ----------
    h5_path:
        Output path.
    y:
        Integer label tensor of shape ``(N,)``.
    Sx:
        Feature tensor of shape ``(N, ...)``.
    params:
        Optional raw-parameter tensor to store alongside.
    """
    import h5py

    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    Sx_np = np.ascontiguousarray(Sx.detach().cpu().numpy())
    y_np = np.ascontiguousarray(y.detach().cpu().numpy()).reshape(-1)
    if not np.issubdtype(y_np.dtype, np.integer):
        raise TypeError("Labels must be integers.")
    y_np = y_np.astype(np.int64, copy=False)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Sx", data=Sx_np, chunks=(1,) + Sx_np.shape[1:],
                         compression="lzf", shuffle=True)
        f.create_dataset("y", data=y_np, chunks=(1,), compression="lzf", shuffle=True)
        f.create_dataset("index", data=np.arange(y_np.shape[0], dtype=np.int64),
                         chunks=(min(1024, max(1, y_np.shape[0])),))
        if params is not None:
            p_np = np.ascontiguousarray(params.detach().cpu().numpy())
            f.create_dataset("params", data=p_np,
                             chunks=(1,) + p_np.shape[1:] if p_np.ndim > 1 else (1,))
        f.attrs["Sx_shape"] = Sx_np.shape
        f.attrs["y_shape"] = y_np.shape


def load_scattering_hdf5(h5_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(y, Sx)`` from an HDF5 cache written by :func:`save_scattering_hdf5`."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        Sx = torch.from_numpy(np.ascontiguousarray(f["Sx"][:]))
        y = torch.from_numpy(np.ascontiguousarray(f["y"][:])).long()
    return y, Sx


class H5ScatteringDataset(Dataset):
    """Memory-light :class:`torch.utils.data.Dataset` backed by an HDF5 file.

    Samples are read lazily from disk, so very large datasets need not fit in
    RAM. Returns ``(x, y)`` pairs, or ``(x, y, params)`` when
    ``with_params=True``.

    Parameters
    ----------
    h5_path:
        Path to a file written by :func:`save_scattering_hdf5`.
    with_params:
        Whether to also return the stored raw parameters.
    """

    def __init__(self, h5_path: str, with_params: bool = False):
        self.h5_path = h5_path
        self.with_params = with_params
        self._f = None
        import h5py

        with h5py.File(self.h5_path, "r") as f:
            self._len = int(f["Sx"].shape[0])

    def __len__(self) -> int:
        return self._len

    def _ensure_open(self):
        if self._f is None:
            import h5py

            self._f = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int):
        self._ensure_open()
        idx = int(idx)
        x = torch.from_numpy(np.ascontiguousarray(self._f["Sx"][idx]))
        y = torch.tensor(int(self._f["y"][idx]), dtype=torch.int64)
        if self.with_params:
            p = torch.from_numpy(np.ascontiguousarray(self._f["params"][idx]))
            return x, y, p
        return x, y

    def __del__(self):
        try:
            if self._f is not None:
                self._f.close()
        except Exception:
            pass
