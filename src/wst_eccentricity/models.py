"""Neural-network classifiers operating on wavelet-scattering features.

The reference model used in the paper is :class:`Conv1DNet`, a compact 1D
convolutional network applied directly to the (flattened) scattering
coefficients. It outputs a single logit per signal for binary
eccentric-vs-quasi-circular classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Conv1DNet"]


class Conv1DNet(nn.Module):
    """Compact 1D convolutional network for binary classification.

    The architecture is intentionally small: a stack of
    ``Conv1d -> BatchNorm -> ReLU -> Dropout`` blocks, followed by global
    average pooling and a two-layer classification head that returns one logit
    per sample (apply :func:`torch.sigmoid` to obtain a probability).

    Parameters
    ----------
    input_size:
        Length of the (flattened) input feature vector. Stored for reference;
        the network itself is fully convolutional and does not depend on it.
    in_channels:
        Number of input channels. Use ``1`` for flattened features, or the
        number of detectors if you feed ``(N, D, L)`` tensors.
    num_filters:
        Number of convolutional filters per layer.
    kernel_size:
        Convolutional kernel size (``"same"`` padding is used).
    num_conv_layers:
        Number of convolutional blocks.
    dropout:
        Dropout probability applied after each block.
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int = 1,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        layers: list[nn.Module] = []
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding="same")
            )
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = num_filters
        self.conv_block = nn.Sequential(*layers)

        self.fc1 = nn.Linear(num_filters, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Either ``(batch, features)`` (treated as a single channel) or
            ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            One logit per sample, shape ``(batch,)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv_block(x)  # (batch, num_filters, length)
        x = x.mean(dim=2)       # global average pooling -> (batch, num_filters)
        x = self.fc2(torch.relu(self.fc1(x)))
        return x.squeeze(-1)    # (batch,)
