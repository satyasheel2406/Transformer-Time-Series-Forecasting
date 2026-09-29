"""
Windowed sequence generation and the PyTorch Dataset wrapper.

Converts a 1D scaled time series into supervised (X, y) pairs where X is a
window of `input_window` past observations and y is the next
`forecast_horizon` future observations, e.g.:

    Input:  [t1, t2, ..., t168]
    Target: [t169, ..., t192]
"""
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from src import config
def create_sequences(
    data: np.ndarray,
    input_window: int = config.INPUT_WINDOW,
    forecast_horizon: int = config.FORECAST_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across a 1D array to build supervised sequences.

    Parameters
    ----------
    data : np.ndarray, shape (n_timesteps,)
        Scaled univariate time series.
    input_window : int
        Number of past timesteps used as input.
    forecast_horizon : int
        Number of future timesteps to predict.

    Returns
    -------
    X : np.ndarray, shape (n_samples, input_window)
    y : np.ndarray, shape (n_samples, forecast_horizon)
    """
    n_timesteps = len(data)
    n_samples = n_timesteps - input_window - forecast_horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough data ({n_timesteps} points) to build even one sequence "
            f"with input_window={input_window} and forecast_horizon={forecast_horizon}."
        )
    X = np.zeros((n_samples, input_window), dtype=np.float32)
    y = np.zeros((n_samples, forecast_horizon), dtype=np.float32)
    for i in range(n_samples):
        X[i] = data[i: i + input_window]
        y[i] = data[i + input_window: i + input_window + forecast_horizon]
    return X, y
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapping windowed (X, y) sequences.

    __getitem__ returns:
        X -> torch.FloatTensor, shape (input_window, 1)   [past observations, 1 feature]
        y -> torch.FloatTensor, shape (forecast_horizon,)  [future observations]

    The trailing feature dimension of 1 on X matches the Transformer's
    expected input shape (batch_size, input_window, num_features), with
    num_features=1 for this univariate setup.
    """
    def __init__(self, data: np.ndarray, input_window: int = config.INPUT_WINDOW,
                 forecast_horizon: int = config.FORECAST_HORIZON):
        self.X, self.y = create_sequences(data, input_window, forecast_horizon)
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).unsqueeze(-1)  # (input_window, 1)
        y = torch.from_numpy(self.y[idx])                # (forecast_horizon,)
        return x, y
