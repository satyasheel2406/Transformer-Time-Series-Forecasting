import random
import time
import numpy as np
import torch
from src import config
from src import data_loader
from src import preprocessing
from src import dataset as dataset_module
from src import transformer_model
from src import baseline as baseline_module
from src import train as train_module
from src import evaluate as evaluate_module
from src import visualization as viz
def set_seeds(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def print_header():
    print("=" * 60)
    print("Transformer Time-Series Forecasting")
    print("=" * 60)
    print()
def main():
    start_time = time.time()
    set_seeds()
    print_header()
    print("Loading dataset...")
    df = data_loader.get_clean_dataframe()
    print("Dataset loaded successfully.\n")
    print("Generating EDA visualizations...")
    viz.plot_full_time_series(df)
    viz.plot_recent_segment(df)
    viz.plot_rolling_statistics(df)
    viz.plot_target_distribution(df)
    viz.plot_seasonal_pattern(df)
    print()
    splits = preprocessing.prepare_splits(df)
    X_train, y_train = dataset_module.create_sequences(splits["train_scaled"])
    X_val, y_val = dataset_module.create_sequences(splits["val_scaled"])
    X_test, y_test = dataset_module.create_sequences(splits["test_scaled"])
    train_dataset = dataset_module.TimeSeriesDataset(splits["train_scaled"])
    val_dataset = dataset_module.TimeSeriesDataset(splits["val_scaled"])
    test_dataset = dataset_module.TimeSeriesDataset(splits["test_scaled"])
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print()
    device = train_module.get_device()
    print(f"Using device: {device}\n")
    model = transformer_model.TimeSeriesTransformer()
    n_params = transformer_model.count_parameters(model)
    print(f"Transformer parameter count: {n_params:,}\n")
    print("Training Transformer...")
    model, history = train_module.train_transformer(model, train_dataset, val_dataset, device=device)
    viz.plot_training_loss(history)
    print()
    transformer_preds_scaled = evaluate_module.get_transformer_predictions(model, test_dataset, device)
    print("Training Linear Regression baseline...")
    lr_baseline = baseline_module.LinearRegressionBaseline()
    lr_baseline.fit(X_train, y_train)
    lr_preds_scaled = lr_baseline.predict(X_test)
    print("Linear Regression baseline trained.\n")
    arima_metrics = None
    try:
        print("Fitting ARIMA baseline on training series...")
        arima = baseline_module.ARIMABaseline(order=(5, 1, 0))
        arima.fit(splits["train_scaled"])
        max_arima_windows = 150
        step = max(1, len(X_test) // max_arima_windows)
        arima_indices = list(range(0, len(X_test), step))[:max_arima_windows]
        arima_preds, arima_true = [], []
        for idx in arima_indices:
            history_window = X_test[idx]
            forecast = arima.predict_from_history(history_window, config.FORECAST_HORIZON)
            if forecast is not None:
                arima_preds.append(forecast)
                arima_true.append(y_test[idx])
        if len(arima_preds) > 0:
            arima_preds = np.array(arima_preds)
            arima_true = np.array(arima_true)
            arima_metrics = evaluate_module.evaluate_predictions(arima_true, arima_preds, splits["scaler"])
            print(f"ARIMA evaluated on {len(arima_preds)} representative test windows.\n")
        else:
            print("ARIMA produced no valid forecasts; excluding from comparison.\n")
    except Exception as exc:
        print(f"ARIMA baseline skipped due to error: {exc}\n")
    print("Evaluating models...\n")

    transformer_metrics = evaluate_module.evaluate_predictions(y_test, transformer_preds_scaled, splits["scaler"])
    lr_metrics = evaluate_module.evaluate_predictions(y_test, lr_preds_scaled, splits["scaler"])

    results = {
        "Linear Regression": lr_metrics,
        "Transformer": transformer_metrics,
    }
    if arima_metrics is not None:
        results["ARIMA"] = arima_metrics
    metrics_df = evaluate_module.save_comparison_csv(results)
    print("Model Performance")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(model_name)
        print(f"MAE  : {metrics['MAE']:.4f}")
        print(f"MAPE : {metrics['MAPE']:.4f}%")
        print()
    print("-" * 40)
    best_model_name = metrics_df.sort_values("MAE").iloc[0]["Model"]
    print(f"\nBest Model: {best_model_name}\n")
    print("Generating forecast visualizations...")
    test_timestamps_full = splits["test_df"][config.TIMESTAMP_COL].values
    sample_idx = len(X_test) // 2  # a representative window from the middle of the test set
    forecast_start = sample_idx + config.INPUT_WINDOW
    forecast_timestamps = test_timestamps_full[forecast_start: forecast_start + config.FORECAST_HORIZON]
    y_true_sample = evaluate_module.inverse_transform(y_test[sample_idx], splits["scaler"])
    transformer_sample = evaluate_module.inverse_transform(transformer_preds_scaled[sample_idx], splits["scaler"])
    lr_sample = evaluate_module.inverse_transform(lr_preds_scaled[sample_idx], splits["scaler"])
    viz.plot_forecast_vs_actual(
        forecast_timestamps, y_true_sample, transformer_sample,
        title="Actual vs Transformer Forecast (24h Horizon)",
        filename="transformer_forecast.png", color="#d62728",
    )
    viz.plot_forecast_vs_actual(
        forecast_timestamps, y_true_sample, lr_sample,
        title="Actual vs Linear Regression Forecast (24h Horizon)",
        filename="baseline_forecast.png", color="#2ca02c",
    )
    viz.plot_model_comparison_bars(metrics_df)
    viz.plot_forecast_zoom(forecast_timestamps, y_true_sample, transformer_sample, lr_sample)
    y_test_orig = evaluate_module.inverse_transform(y_test, splits["scaler"])
    transformer_preds_orig = evaluate_module.inverse_transform(transformer_preds_scaled, splits["scaler"])
    lr_preds_orig = evaluate_module.inverse_transform(lr_preds_scaled, splits["scaler"])
    transformer_mae_by_step = np.mean(np.abs(y_test_orig - transformer_preds_orig), axis=0)
    lr_mae_by_step = np.mean(np.abs(y_test_orig - lr_preds_orig), axis=0)
    viz.plot_error_by_hour(transformer_mae_by_step, "Transformer", "transformer_error_by_horizon.png")
    viz.plot_error_by_hour(lr_mae_by_step, "Linear Regression", "baseline_error_by_horizon.png")
    print("\nError Analysis")
    print("-" * 40)
    print(f"Transformer MAE at hour+1  : {transformer_mae_by_step[0]:.2f}")
    print(f"Transformer MAE at hour+24 : {transformer_mae_by_step[-1]:.2f}")
    print(f"Linear Reg. MAE at hour+1  : {lr_mae_by_step[0]:.2f}")
    print(f"Linear Reg. MAE at hour+24 : {lr_mae_by_step[-1]:.2f}")
    print("-" * 40)
    elapsed = time.time() - start_time
    print(f"\nTotal pipeline runtime: {elapsed / 60:.1f} minutes")
    print("\nDone. See results/figures/ and results/metrics/model_comparison.csv")
if __name__ == "__main__":
    main()
