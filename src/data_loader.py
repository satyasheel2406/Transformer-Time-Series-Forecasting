"""
Data loading utilities.
Responsible for reading the raw CSV, standardizing column names, parsing
timestamps, sorting chronologically, and reporting basic dataset health
(missing values, date range, size) before any modeling happens.
"""
import os
import pandas as pd
from src import config
def load_raw_dataset(csv_path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw energy-consumption CSV and standardize it into a
    two-column DataFrame: [timestamp, target_value].
    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file (see data/README.md for where to get it).
    Returns
    -------
    pd.DataFrame
        Chronologically sorted DataFrame with columns
        [config.TIMESTAMP_COL, config.TARGET_COL].
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset at '{csv_path}'.\n"
            f"See data/README.md for download instructions."
        )

    df = pd.read_csv(csv_path)
    # The raw AEP_hourly.csv file has columns: Datetime, AEP_MW
    # We standardize to generic names so the rest of the pipeline is
    # dataset-agnostic (any two-column [timestamp, value] CSV would work).
    if "Datetime" in df.columns and "AEP_MW" in df.columns:
        df = df.rename(columns={"Datetime": config.TIMESTAMP_COL, "AEP_MW": config.TARGET_COL})
    elif config.TIMESTAMP_COL in df.columns and config.TARGET_COL in df.columns:
        pass  # already standardized
    else:
        # Fall back: assume first column is timestamp, second is the target.
        original_cols = df.columns.tolist()
        df = df.rename(columns={original_cols[0]: config.TIMESTAMP_COL,
                                 original_cols[1]: config.TARGET_COL})
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
    df = df[[config.TIMESTAMP_COL, config.TARGET_COL]]
    # Sort chronologically and drop exact duplicate timestamps (keep first).
    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
    df = df.drop_duplicates(subset=config.TIMESTAMP_COL, keep="first").reset_index(drop=True)
    return df
def restrict_to_recent_window(df: pd.DataFrame, n_hours: int = config.N_HOURS_USED) -> pd.DataFrame:
    """
    Keep only the most recent `n_hours` rows of a chronologically sorted
    DataFrame. Keeps the project runnable on a laptop while still using
    real, unmodified sensor readings.
    """
    if len(df) <= n_hours:
        return df.reset_index(drop=True)
    return df.iloc[-n_hours:].reset_index(drop=True)
def report_dataset_info(df: pd.DataFrame) -> None:
    """Print a concise health report of the loaded dataset."""
    n_missing = df[config.TARGET_COL].isna().sum()
    n_rows = len(df)
    print("Dataset Information")
    print("-" * 40)
    print(f"Rows                : {n_rows}")
    print(f"Date range          : {df[config.TIMESTAMP_COL].min()} -> {df[config.TIMESTAMP_COL].max()}")
    print(f"Missing target rows : {n_missing} ({100 * n_missing / max(n_rows, 1):.3f}%)")
    print(f"Target min / max    : {df[config.TARGET_COL].min():.2f} / {df[config.TARGET_COL].max():.2f}")
    print(f"Target mean / std   : {df[config.TARGET_COL].mean():.2f} / {df[config.TARGET_COL].std():.2f}")
    print("-" * 40)
def get_clean_dataframe() -> pd.DataFrame:
    """
    Convenience entry point used by main.py: load, restrict to a laptop-
    friendly recent window, report info, and return a clean DataFrame.
    Any missing target values are linearly interpolated (still real
    measured data around each gap, no fabricated readings).
    """
    df = load_raw_dataset()
    df = restrict_to_recent_window(df)
    n_missing_before = df[config.TARGET_COL].isna().sum()
    if n_missing_before > 0:
        df[config.TARGET_COL] = df[config.TARGET_COL].interpolate(method="linear")
        df[config.TARGET_COL] = df[config.TARGET_COL].bfill().ffill()
    report_dataset_info(df)
    return df
