"""
Preprocessing: chronological train/val/test split and scaling.

CRITICAL leakage-prevention rules enforced here:
  1. The split is purely chronological (train is the earliest slice, test
     is the latest slice). We never shuffle before splitting a time series.
  2. The scaler is fit ONLY on the training slice, then reused (transform
     only, never re-fit) on validation and test.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import config
def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train/val/test slices by position,
    not randomly. Random splitting would let the model "see the future"
    during training (temporal leakage) because adjacent time steps are
    highly correlated - a randomly-split validation/test point could sit
    directly next to a training point in time, making performance look far
    better than it would in real deployment.
    """
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df
def fit_scaler_on_train(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler using ONLY the training target values."""
    scaler = StandardScaler()
    scaler.fit(train_df[[config.TARGET_COL]].values)
    return scaler
def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform (never re-fit) a DataFrame's target column with a fitted scaler."""
    return scaler.transform(df[[config.TARGET_COL]].values).flatten()
def prepare_splits(df: pd.DataFrame):
    """
    Full preprocessing pipeline: chronological split -> fit scaler on train
    only -> transform all three splits.
"""
    train_df, val_df, test_df = chronological_split(df)

    scaler = fit_scaler_on_train(train_df)

    train_scaled = apply_scaler(train_df, scaler)
    val_scaled = apply_scaler(val_df, scaler)
    test_scaled = apply_scaler(test_df, scaler)

    print("Chronological Split")
    print("-" * 40)
    print(f"Train samples : {len(train_df)}  ({train_df[config.TIMESTAMP_COL].min()} -> {train_df[config.TIMESTAMP_COL].max()})")
    print(f"Val samples   : {len(val_df)}  ({val_df[config.TIMESTAMP_COL].min()} -> {val_df[config.TIMESTAMP_COL].max()})")
    print(f"Test samples  : {len(test_df)}  ({test_df[config.TIMESTAMP_COL].min()} -> {test_df[config.TIMESTAMP_COL].max()})")
    print("-" * 40)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
    }
