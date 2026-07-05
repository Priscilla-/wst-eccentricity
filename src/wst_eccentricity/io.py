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
    "match_waveform_and_parameter_files",
    "load_hdf5_data",
    "read_parameters",
    "load_scattering_data",
]

#: Sentinel returned by :func:`get_seed_and_index` for unparsable file names.
_UNMATCHED = (9999, 999999)


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
    return _UNMATCHED


def match_waveform_and_parameter_files(
    waveforms_dir: str,
    params_dir: str,
    waveform_pattern: str = "*.hdf5",
    params_pattern: str = "*.txt",
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Pair waveform and parameter files by their ``s<seed>_<index>`` tag.

    Waveform files (``waveform_s42_7_*.hdf5``) and parameter files
    (``params_s42_7.txt``) that share the same ``(seed, index)`` tag describe
    the same signal. Pairing by tag -- rather than trusting that two
    independently sorted directory listings line up -- guarantees that
    features and labels stay aligned even if a file is missing, duplicated,
    or a stray file is present in either folder.

    Parameters
    ----------
    waveforms_dir:
        Directory containing the waveform ``.hdf5`` files.
    params_dir:
        Directory containing the ``params_*.txt`` files.
    waveform_pattern, params_pattern:
        Glob patterns selecting the files in each directory.
    strict:
        If ``True`` (default), raise when any file in one folder has no
        counterpart in the other. If ``False``, drop unmatched files with a
        printed warning and continue with the intersection.

    Returns
    -------
    waveform_files, parameter_files : list of str
        Two equally long lists, sorted by ``(seed, index)``, such that
        ``waveform_files[i]`` and ``parameter_files[i]`` refer to the same
        signal.
    """
    wf_files = glob.glob(os.path.join(waveforms_dir, waveform_pattern))
    pf_files = glob.glob(os.path.join(params_dir, params_pattern))
    if not wf_files:
        raise FileNotFoundError(f"No {waveform_pattern} files found in {waveforms_dir!r}")
    if not pf_files:
        raise FileNotFoundError(f"No {params_pattern} files found in {params_dir!r}")

    def tag_map(files: list[str], what: str) -> dict[tuple[int, int], str]:
        mapping: dict[tuple[int, int], str] = {}
        for path in files:
            tag = get_seed_and_index(path)
            if tag == _UNMATCHED:
                print(f"  warning: skipping {what} file without s<seed>_<index> tag: "
                      f"{os.path.basename(path)}")
                continue
            if tag in mapping:
                raise ValueError(
                    f"Duplicate tag s{tag[0]}_{tag[1]} among {what} files: "
                    f"{os.path.basename(mapping[tag])} and {os.path.basename(path)}"
                )
            mapping[tag] = path
        return mapping

    wmap = tag_map(wf_files, "waveform")
    pmap = tag_map(pf_files, "parameter")

    common = sorted(set(wmap) & set(pmap))
    only_w = sorted(set(wmap) - set(pmap))
    only_p = sorted(set(pmap) - set(wmap))
    if only_w or only_p:
        msg = (
            f"{len(only_w)} waveform file(s) without parameters "
            f"(e.g. {only_w[:3]}) and {len(only_p)} parameter file(s) without "
            f"waveforms (e.g. {only_p[:3]})."
        )
        if strict:
            raise ValueError(
                "Waveform/parameter mismatch: " + msg +
                " Fix the dataset, or call with strict=False to use the "
                f"{len(common)} matched pairs."
            )
        print(f"  warning: {msg} Continuing with {len(common)} matched pairs.")
    if not common:
        raise ValueError(
            f"No matching s<seed>_<index> tags between {waveforms_dir!r} and {params_dir!r}."
        )
    return [wmap[t] for t in common], [pmap[t] for t in common]


def load_hdf5_data(
    folder: str | None = None,
    np_dtype: type = np.float32,
    sort_by_seed: bool = True,
    start_from: int = 0,
    files: list[str] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Load every ``.hdf5`` waveform file in ``folder`` into one tensor.

    Each file is assumed to contain a single dataset, all of the same shape
    (for example ``(n_detectors, n_samples)``).

    Parameters
    ----------
    folder:
        Directory containing the ``.hdf5`` files (ignored if ``files`` is given).
    np_dtype:
        NumPy dtype used for the pre-allocated output array.
    sort_by_seed:
        If ``True`` (default), sort files by their ``(seed, index)`` tag so the
        ordering matches :func:`read_parameters`. If ``False``, sort
        alphabetically. Ignored if ``files`` is given (its order is kept).
    start_from:
        Skip the first ``start_from`` files (useful for resuming).
    files:
        Explicit list of file paths to load, e.g. produced by
        :func:`match_waveform_and_parameter_files`. Takes precedence over
        ``folder``.

    Returns
    -------
    data : torch.Tensor
        Tensor of shape ``(n_files, *signal_shape)``.
    files : list of str
        The file paths, in the same order as ``data``.
    """
    if files is None:
        if folder is None:
            raise ValueError("Provide either `folder` or an explicit `files` list.")
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
    folder: str | None = None,
    pattern: str = "*.txt",
    verbose: bool = False,
    files: list[str] | None = None,
) -> list[dict[str, float]]:
    """Read the plain-text parameter files in ``folder``.

    Each file is expected to contain ``key: value`` lines, e.g.::

        eccentricity: 0.023
        mass_1: 35.0
        NSNR: 42

    Parameters
    ----------
    folder:
        Directory containing the parameter files (ignored if ``files`` is given).
    pattern:
        Glob pattern selecting the files (default ``"*.txt"``).
    verbose:
        If ``True``, print the first few file names for a sanity check.
    files:
        Explicit list of file paths to read, e.g. produced by
        :func:`match_waveform_and_parameter_files`. Takes precedence over
        ``folder`` (its order is kept).

    Returns
    -------
    list of dict
        One ``{parameter_name: float_value}`` dictionary per file, ordered by
        the ``(seed, index)`` tag so the ordering matches :func:`load_hdf5_data`.
        Non-numeric values (e.g. an approximant name) are skipped with a
        warning rather than raising.
    """
    if files is None:
        if folder is None:
            raise ValueError("Provide either `folder` or an explicit `files` list.")
        files = sorted(glob.glob(os.path.join(folder, pattern)), key=get_seed_and_index)
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in: {folder!r}")
    if verbose:
        print("First parameter files:", files[:4])

    warned_keys: set[str] = set()
    records: list[dict[str, float]] = []
    for path in files:
        record: dict[str, float] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                try:
                    record[key] = float(val.strip())
                except ValueError:
                    if key not in warned_keys:
                        warned_keys.add(key)
                        print(f"  warning: skipping non-numeric parameter "
                              f"{key!r} (e.g. {val.strip()!r})")
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
