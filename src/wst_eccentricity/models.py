"""Neural-network classifier operating on wavelet-scattering features.

The reference model used in the paper is :class:`SWT_CNN_1D_Binned`, a compact
1D convolutional network that consumes the *native* wavelet-scattering tensor of
shape ``(batch, detectors, channels, time)`` -- i.e. the detector, scattering-
channel and time axes are kept separate rather than flattened into one long
vector. It outputs a single logit per signal for binary eccentric-vs-quasi-
circular classification (apply :func:`torch.sigmoid` to get a probability).

Why this shape matters
----------------------
Each gravitational-wave signal is observed by ``D`` detectors. For every
detector the wavelet scattering transform produces ``C`` scattering channels
sampled at ``T`` time steps. Keeping these axes separate lets the network

* run a shared 1D CNN **along time** for each detector (so it learns which
  scattering channels co-vary in time around the merger), and
* combine the detectors afterwards as an unordered *set* using simple pooling
  statistics -- which keeps the model invariant to detector ordering.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SWT_CNN_1D_Binned"]


class SWT_CNN_1D_Binned(nn.Module):
    """Binned 1D-CNN for binary eccentricity classification.

    The network processes each detector independently with a shared 1D CNN,
    compresses the time axis into a fixed number of coarse, merger-aligned
    ``time_bins``, then fuses the detectors with order-invariant pooling
    statistics before a small classification head.

    Expected input shape is ``(batch, num_detectors, in_channels, time)``.

    Parameters
    ----------
    in_channels:
        Number of scattering channels ``C`` per detector (the input channels of
        the first convolution).
    num_detectors:
        Number of detectors ``D`` in each sample.
    dropout_rate:
        Dropout probability used after detector embedding and inside the head.
    cnn_channels:
        Output channels of the three convolutional blocks, ``(c1, c2, c3)``.
    kernel_sizes:
        Kernel sizes of the three convolutional blocks, ``(k1, k2, k3)``.
        ``"same"``-style padding (``k // 2``) keeps the time length until
        pooling.
    time_bins:
        Number ``K`` of coarse time segments the signal is reduced to via
        adaptive average pooling. Because the waveforms are merger-aligned, each
        bin corresponds to a consistent phase of the inspiral+merger.

    Notes
    -----
    ``forward`` returns one logit per sample with shape ``(batch,)``, matching
    the convention expected by :func:`wst_eccentricity.training.train_binary`
    and :class:`torch.nn.BCEWithLogitsLoss`.
    """

    def __init__(
        self,
        in_channels: int = 56,
        num_detectors: int = 3,
        dropout_rate: float = 0.3,
        cnn_channels: tuple[int, int, int] = (32, 32, 32),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
        time_bins: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_detectors = num_detectors
        self.time_bins = time_bins

        c1, c2, c3 = cnn_channels
        k1, k2, k3 = kernel_sizes
        self.det_feat_dim = c3 * time_bins  # per-detector feature length (C3 * K)

        # Per-detector feature extractor. Shared across detectors so the model
        # learns one detector-agnostic set of time-local filters.
        self.detectors_cnn = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # halves the time length

            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # halves again

            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )

        # Because waveforms are merger-aligned, this pooling is structure-aware:
        # it summarises the time evolution into ``K`` coarse, comparable segments.
        self.time_pool = nn.AdaptiveAvgPool1d(time_bins)  # (B, C3, L) -> (B, C3, K)
        self.dropout = nn.Dropout(dropout_rate)

        # Fusion head. ``det_pool`` produces 3 pooling statistics per feature,
        # so the head input is ``3 * det_feat_dim``.
        pooled_dim = 3 * self.det_feat_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),  # single logit
        )

    def det_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Combine per-detector embeddings with order-invariant statistics.

        Parameters
        ----------
        x:
            Detector embeddings of shape ``(B, D, F)`` where ``F`` is the
            per-detector feature length.

        Returns
        -------
        torch.Tensor
            Concatenated statistics ``[energy, max, mean]`` over the detector
            axis, shape ``(B, 3 * F)``. Pooling over detectors makes the result
            invariant to the ordering (and number) of detectors.
        """
        mean = x.mean(dim=1)            # average detector response
        mx = x.max(dim=1).values       # strongest detector response
        energy = (x ** 2).sum(dim=1)   # total "energy" across detectors
        return torch.cat([energy, mx, mean], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Scattering features of shape
            ``(batch, num_detectors, in_channels, time)``.

        Returns
        -------
        torch.Tensor
            One logit per sample, shape ``(batch,)``.
        """
        B, D, C, T = x.shape

        # Embed each detector separately with the shared CNN, then bin its time
        # axis down to ``K`` coarse segments.
        det_embeddings = []
        for d in range(D):
            xi = x[:, d, :, :]                # (B, C, T)
            feat = self.detectors_cnn(xi)     # (B, C3, L)
            feat = self.time_pool(feat)       # (B, C3, K)
            det_embeddings.append(feat.flatten(1))  # (B, C3 * K)

        z = torch.stack(det_embeddings, dim=1)  # (B, D, C3 * K)
        z = self.dropout(z)

        pooled = self.det_pool(z)   # (B, 3 * C3 * K)
        logits = self.head(pooled)  # (B, 1)
        return logits.squeeze(-1)   # (B,)

    def info(self, print_details: bool = False) -> None:
        """Print a short summary of the model and its trainable parameter count.

        Parameters
        ----------
        print_details:
            If ``True``, also print the full module repr.
        """
        print("Model: SWT_CNN_1D_Binned")
        if print_details:
            print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}", flush=True)
