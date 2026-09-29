"""
Classical baselines: Linear Regression (lag features) and ARIMA.

These exist to answer one honest question: does the Transformer actually
buy us anything over a much simpler, much cheaper model? Neither baseline
is artificially handicapped - the Linear Regression baseline sees the same
input window (t-168 ... t-1) and predicts the same horizon (t ... t+23) as
the Transformer, evaluated on the exact same test sequences.
"""

from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression


def build_lag_features(X_windows: np.ndarray) -> np.ndarray:
    """
    X_windows: (n_samples, input_window) scaled values already produced by
    create_sequences(). Linear Regression treats every past timestep in the
    window as one lag feature: t-168, t-167, ..., t-1.
    """
    return X_windows.reshape(X_windows.shape[0], -1)


class LinearRegressionBaseline:
    """
    Multi-output Linear Regression baseline: predicts the full forecast
    horizon in one shot from lag features, using scikit-learn's native
    multi-output support (fits one linear model per horizon step, sharing
    the same lag-feature matrix).
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearRegressionBaseline":
        lag_features = build_lag_features(X_train)
        self.model.fit(lag_features, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        lag_features = build_lag_features(X)
        return self.model.predict(lag_features)


class ARIMABaseline:
    """
    ARIMA baseline fit on the training portion of the (scaled) series.
    For each test sequence we refit-free, rolling forecast is too slow for
    a laptop over thousands of test windows, so we instead fit ARIMA once
    on the training series and roll it forward using statsmodels' `apply`
    to append test history without re-estimating parameters, forecasting
    `forecast_horizon` steps ahead for each test window's cutoff point.
    """

    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.fitted_model = None

    def fit(self, train_series: np.ndarray):
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train_series, order=self.order)
        self.fitted_model = model.fit()
        return self

    def predict_from_history(self, history: np.ndarray, forecast_horizon: int) -> Optional[np.ndarray]:
        """
        Given a history array (the same values used as the Transformer's
        input window, scaled), append it to the fitted ARIMA model's state
        (without re-estimating parameters) and forecast forward.
        """
        try:
            extended = self.fitted_model.apply(history)
            forecast = extended.forecast(steps=forecast_horizon)
            return np.asarray(forecast, dtype=np.float32)
        except Exception:
            return None
