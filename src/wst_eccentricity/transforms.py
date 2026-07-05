"""Wavelet scattering transform (WST) of gravitational-wave strain data.

This module wraps `Kymatio <https://www.kymatio.io>`_'s :class:`Scattering1D`
to turn whitened, merger-aligned strain time series into compact,
translation-invariant scattering coefficients that are then fed to the
classifiers in :mod:`wst_eccentricity.models`.

Only two hyper-parameters control the transform:

``J``
    The maximum scattering scale; the averaging window spans ``2**J`` samples.
``Q``
    The number of wavelets per octave (frequency resolution).
"""

from __future__ import annotations

import os

import torch

__all__ = ["compute_scattering", "scatter_in_batches"]


def compute_scattering(
    gws: torch.Tensor,
    J: int,
    Q: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Compute the 1D wavelet scattering transform of a batch of signals.

    Parameters
    ----------
    gws:
        Real-valued strain tensor of shape ``(N, D, T)`` -- ``N`` signals, each
        recorded at ``D`` detectors with ``T`` time samples. The same ``J`` and
        ``Q`` are used for every detector.
    J:
        Maximum scattering scale (averaging window ``2**J`` samples).
    Q:
        Number of wavelets per octave.
    device:
        Device on which to run the transform (``"cpu"`` or ``"cuda"``).
    dtype:
        Floating-point dtype for the computation.

    Returns
    -------
    Sx : torch.Tensor
        Scattering coefficients of shape ``(N, D, C, T_scatter)``, where ``C``
        is the number of scattering channels and ``T_scatter`` the number of
        (sub-sampled) time bins.
    meta : dict
        Kymatio metadata (``order``, ``xi``, ``sigma``, ...) describing each of
        the ``C`` channels. Identical for every detector.

    Notes
    -----
    Kymatio is imported lazily so that ``import wst_eccentricity`` succeeds even
    in environments (e.g. documentation builds) where Kymatio is not installed.
    """
    from kymatio.torch import Scattering1D

    if gws.dim() != 3:
        raise ValueError(f"Expected gws of shape (N, D, T); got {tuple(gws.shape)}")

    N, D, T = gws.shape
    gws = gws.to(device=device, dtype=dtype).contiguous()

    scattering = Scattering1D(J=J, shape=T, Q=Q).to(device)
    meta = scattering.meta()  # same for all detectors

    x = gws.view(N * D, T)                # (N*D, T)
    Sx = scattering(x).detach()           # (N*D, C, T_scatter)
    Sx = Sx.view(N, D, *Sx.shape[1:])     # (N, D, C, T_scatter)
    return Sx, meta


def scatter_in_batches(
    gws: torch.Tensor,
    J: int,
    Q: int,
    out_dir: str,
    batch_size: int = 10_000,
    device: str = "cpu",
    names: list[str] | None = None,
) -> list[str]:
    """Scatter a large set of signals in batches and save the results to disk.

    Coefficients are written as ``WST_{J}_{Q}_{batch}.pt`` files under
    ``out_dir``; each file stores a dict with keys ``"Sx"``, ``"config"`` and
    ``"meta"`` and can be read back with
    :func:`wst_eccentricity.io.load_scattering_data`.

    Parameters
    ----------
    gws:
        Strain tensor of shape ``(N, D, T)``.
    J, Q:
        Scattering hyper-parameters (see :func:`compute_scattering`).
    out_dir:
        Directory to write the ``.pt`` files into (created if needed).
    batch_size:
        Number of signals processed per batch.
    device:
        Device on which to run the transform.
    names:
        Optional list of ``len(gws)`` identifiers (e.g. waveform file base
        names), stored in each batch's ``config["files"]``. This lets
        consumers (e.g. :func:`wst_eccentricity.pipeline.run_pipeline`) verify
        that a cached transform still corresponds to the current dataset.

    Returns
    -------
    list of str
        Paths of the files that were written, in order.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = gws.shape[0]
    if names is not None and len(names) != n:
        raise ValueError(f"len(names)={len(names)} does not match len(gws)={n}")
    n_batches = (n + batch_size - 1) // batch_size
    written: list[str] = []

    for b in range(n_batches):
        lo, hi = b * batch_size, min((b + 1) * batch_size, n)
        chunk = gws[lo:hi]
        Sx, meta = compute_scattering(chunk, J, Q, device=device)
        cfg = {"J": J, "Q": Q, "T": gws.shape[2], "average": True, "batch": b + 1}
        if names is not None:
            cfg["files"] = list(names[lo:hi])
        path = os.path.join(out_dir, f"WST_{J}_{Q}_{b + 1}.pt")
        torch.save({"Sx": Sx.cpu(), "config": cfg, "meta": meta}, path)
        written.append(path)

    return written
