import os
import matplotlib
matplotlib.use("Agg")  # headless / script-safe backend, no notebook required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src import config
sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True
def _savefig(fig, filename: str):
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
# EDA plots
def plot_full_time_series(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[config.TIMESTAMP_COL], df[config.TARGET_COL], linewidth=0.6, color="#1f77b4")
    ax.set_title("Full Time Series: Hourly Energy Consumption (AEP, MW)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Energy Consumption (MW)")
    _savefig(fig, "time_series.png")


def plot_recent_segment(df: pd.DataFrame, n_hours: int = 24 * 14):
    recent = df.iloc[-n_hours:]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(recent[config.TIMESTAMP_COL], recent[config.TARGET_COL], color="#d62728")
    ax.set_title(f"Recent Segment: Last {n_hours} Hours ({n_hours // 24} Days)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Energy Consumption (MW)")
    _savefig(fig, "recent_segment.png")


def plot_rolling_statistics(df: pd.DataFrame, window: int = 24 * 7):
    rolling_mean = df[config.TARGET_COL].rolling(window=window).mean()
    rolling_std = df[config.TARGET_COL].rolling(window=window).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df[config.TIMESTAMP_COL], df[config.TARGET_COL], alpha=0.3, label="Original", color="gray")
    axes[0].plot(df[config.TIMESTAMP_COL], rolling_mean, label=f"Rolling Mean ({window}h)", color="#1f77b4")
    axes[0].set_title("Rolling Mean (7-Day Window)")
    axes[0].set_ylabel("Energy Consumption (MW)")
    axes[0].legend()

    axes[1].plot(df[config.TIMESTAMP_COL], rolling_std, color="#ff7f0e", label=f"Rolling Std ({window}h)")
    axes[1].set_title("Rolling Standard Deviation (7-Day Window)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Std Dev (MW)")
    axes[1].legend()

    _savefig(fig, "rolling_statistics.png")


def plot_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[config.TARGET_COL], bins=60, kde=True, ax=ax, color="#2ca02c")
    ax.set_title("Distribution of Hourly Energy Consumption")
    ax.set_xlabel("Energy Consumption (MW)")
    ax.set_ylabel("Frequency")
    _savefig(fig, "distribution.png")


def plot_seasonal_pattern(df: pd.DataFrame):
    tmp = df.copy()
    tmp["hour"] = tmp[config.TIMESTAMP_COL].dt.hour
    tmp["dayofweek"] = tmp[config.TIMESTAMP_COL].dt.dayofweek

    hourly_avg = tmp.groupby("hour")[config.TARGET_COL].mean()
    dow_avg = tmp.groupby("dayofweek")[config.TARGET_COL].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hourly_avg.index, hourly_avg.values, marker="o", color="#9467bd")
    axes[0].set_title("Average Consumption by Hour of Day")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Avg Energy Consumption (MW)")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(day_labels, dow_avg.values, color="#17becf")
    axes[1].set_title("Average Consumption by Day of Week")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Avg Energy Consumption (MW)")

    _savefig(fig, "seasonal_patterns.png")
# Training curve
def plot_training_loss(history: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    ax.plot(epochs, history["val_loss"], label="Validation Loss", color="#d62728")
    ax.set_title("Transformer Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled units)")
    ax.legend()
    _savefig(fig, "training_loss.png")
# Forecast comparison plots
def plot_forecast_vs_actual(timestamps, y_true, y_pred, title: str, filename: str, color: str = "#d62728"):
    """
    y_true / y_pred: 1D arrays of the same length as `timestamps`, aligned
    to the forecast horizon of a single representative test window.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, y_true, label="Actual", color="#1f77b4", marker="o", markersize=3)
    ax.plot(timestamps, y_pred, label="Predicted", color=color, marker="x", markersize=4, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend()
    fig.autofmt_xdate()
    _savefig(fig, filename)


def plot_model_comparison_bars(metrics_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(metrics_df["Model"], metrics_df["MAE"], color="#1f77b4")
    axes[0].set_title("MAE Comparison (Lower is Better)")
    axes[0].set_ylabel("MAE (MW)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metrics_df["Model"], metrics_df["MAPE"], color="#ff7f0e")
    axes[1].set_title("MAPE Comparison (Lower is Better)")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].tick_params(axis="x", rotation=20)

    _savefig(fig, "model_comparison.png")


def plot_forecast_zoom(timestamps, y_true, y_transformer, y_baseline, filename: str = "forecast_zoom.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, y_true, label="Actual", color="black", linewidth=2)
    ax.plot(timestamps, y_transformer, label="Transformer", color="#d62728", linestyle="--", marker="x", markersize=4)
    ax.plot(timestamps, y_baseline, label="Linear Regression Baseline", color="#2ca02c", linestyle="--", marker="s", markersize=3)
    ax.set_title("Forecast Zoom: Transformer vs Baseline vs Actual (24h Horizon)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend()
    fig.autofmt_xdate()
    _savefig(fig, filename)


def plot_error_by_hour(errors_by_horizon_step: np.ndarray, model_name: str, filename: str):
    """Bar chart of mean absolute error at each step of the forecast horizon."""
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(1, len(errors_by_horizon_step) + 1)
    ax.bar(steps, errors_by_horizon_step, color="#e377c2")
    ax.set_title(f"{model_name}: Mean Absolute Error by Forecast Step")
    ax.set_xlabel("Hours Ahead")
    ax.set_ylabel("MAE (MW)")
    _savefig(fig, filename)
