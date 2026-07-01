"""Training loop for the binary eccentricity classifier.

A single, self-contained training function with early stopping on the
validation AUC. It is deliberately framework-light (plain PyTorch) so it is
easy to read and adapt.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .metrics import auc_ap

__all__ = ["train_binary"]


def train_binary(
    model: nn.Module,
    train_loader,
    val_loader,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 5,
    device: str = "cpu",
):
    """Train a binary classifier with early stopping on validation AUC.

    The best weights (highest validation AUC, ties broken by lower validation
    loss) are restored into ``model`` before returning.

    Parameters
    ----------
    model:
        A model returning one logit per sample. Trained in place.
    train_loader, val_loader:
        Data loaders yielding ``(x, y)`` batches.
    lr:
        Adam learning rate.
    max_epochs:
        Maximum number of epochs.
    patience:
        Stop after this many epochs without improvement in validation AUC.
    device:
        Device to train on (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    model : torch.nn.Module
        The model with the best weights restored.
    history : dict
        Lists of ``"train_loss"``, ``"val_loss"`` and ``"val_auc"`` per epoch.
    best_val_auc : float
        The best validation AUC reached.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_state = None
    best_val_auc = -np.inf
    best_val_loss = np.inf
    epochs_no_improve = 0

    for _epoch in range(1, max_epochs + 1):
        # --- training ---
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().view(-1)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        # --- validation ---
        model.eval()
        val_running, val_n = 0.0, 0
        targets, probs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).float().view(-1)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running += loss.item() * xb.size(0)
                val_n += xb.size(0)
                targets.append(yb.cpu().numpy())
                probs.append(torch.sigmoid(logits).cpu().numpy())
        val_loss = val_running / max(val_n, 1)
        val_auc, _ = auc_ap(np.concatenate(targets), np.concatenate(probs))

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_auc"].append(float(val_auc))

        improved = (val_auc > best_val_auc) or (
            val_auc == best_val_auc and val_loss < best_val_loss
        )
        if improved:
            best_val_auc = float(val_auc)
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, float(best_val_auc)
