"""
Central configuration for the Transformer time-series forecasting project.
All tunable parameters live here so that the rest of the codebase never
hardcodes a magic number. Nothing in this file touches the test set, so
changing these values does not create any data leakage risk on its own -
leakage protection is enforced structurally in preprocessing.py and
dataset.py (scaler fit only on train, chronological split only).
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
RAW_DATA_PATH = os.path.join(DATA_DIR, "AEP_hourly.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_transformer.pt")
METRICS_CSV_PATH = os.path.join(METRICS_DIR, "model_comparison.csv")
TIMESTAMP_COL = "timestamp"
TARGET_COL = "target_value"
N_HOURS_USED = 10000  # ~14 months of hourly data
INPUT_WINDOW = 168       # 7 days of hourly history
FORECAST_HORIZON = 24    # predict the next 24 hours
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
# Training
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 4  # early stopping patience, measured in epochs without val improvement
# Chronological split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# Reproducibility
RANDOM_SEED = 42
# Misc
EPSILON = 1e-8  # used for safe MAPE computation
