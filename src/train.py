"""
Training loop for the Transformer forecaster: train/validation loop, Adam
optimizer, early stopping on validation loss, best-checkpoint saving, and
CPU/GPU device detection.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_transformer(model: nn.Module, train_dataset, val_dataset, device=None, verbose: bool = True):
    """
    Train `model` with Adam + MSE loss, early stopping on validation loss,
    and checkpoint the best-performing weights to config.BEST_MODEL_PATH.

    Returns
    -------
    model : nn.Module
        Model loaded with the best (lowest val loss) weights.
    history : dict
        {"train_loss": [...], "val_loss": [...]}
    """
    device = device or get_device()
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # Training 
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start
        if verbose:
            print(f"Epoch {epoch}/{config.EPOCHS}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}  ({epoch_time:.1f}s)")
            print()

        # Early stopping / checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state_dict, config.BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE:
                if verbose:
                    print(f"Early stopping triggered after {epoch} epochs "
                          f"(no val improvement for {config.PATIENCE} epochs).")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history
