"""
Evaluation utilities: safe MAE / MAPE computation on the original
(inverse-transformed) scale, and orchestration of the Transformer vs.
baseline comparison on the exact same test period.
"""
from typing import Dict
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src import config
def inverse_transform(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform scaled values back to the original target units."""
    flat = values.reshape(-1, 1)
    original = scaler.inverse_transform(flat)
    return original.reshape(values.shape)
def mean_absolute_error_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def mean_absolute_percentage_error_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safe MAPE: guards against division-by-zero when y_true contains zeros
    (or values extremely close to zero) by excluding those points from the
    percentage-error average rather than letting them blow up to infinity.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mask = np.abs(y_true) > config.EPSILON
    if mask.sum() == 0:
        return float("nan")

    pct_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct_errors) * 100.0)
@torch.no_grad()
def get_transformer_predictions(model, dataset, device) -> np.ndarray:
    """Run the trained Transformer over an entire dataset, batched, no grad."""
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)
def evaluate_predictions(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray,
                          scaler: StandardScaler) -> Dict[str, float]:
    """
    Inverse-transform both true and predicted values back to the original
    scale, then compute MAE and MAPE on that original scale (not on the
    scaled/normalized values, which would be meaningless in real units).
    """
    y_true_orig = inverse_transform(y_true_scaled, scaler)
    y_pred_orig = inverse_transform(y_pred_scaled, scaler)

    mae = mean_absolute_error_safe(y_true_orig, y_pred_orig)
    mape = mean_absolute_percentage_error_safe(y_true_orig, y_pred_orig)

    return {"MAE": mae, "MAPE": mape}
def save_comparison_csv(results: Dict[str, Dict[str, float]], path: str = config.METRICS_CSV_PATH) -> pd.DataFrame:
    """Save a Model,MAE,MAPE comparison table."""
    rows = []
    for model_name, metrics in results.items():
        rows.append({"Model": model_name, "MAE": metrics["MAE"], "MAPE": metrics["MAPE"]})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
