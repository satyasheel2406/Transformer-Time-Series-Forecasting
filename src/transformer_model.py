import math
import torch
import torch.nn as nn
from src import config
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-only Transformer for direct multi-step forecasting.

    A single forward pass predicts the entire forecast_horizon at once
    (rather than autoregressively), which is simpler, faster to train, and
    avoids compounding one-step errors.
    """

    def __init__(
        self,
        input_window: int = config.INPUT_WINDOW,
        forecast_horizon: int = config.FORECAST_HORIZON,
        num_features: int = 1,
        d_model: int = config.D_MODEL,
        nhead: int = config.N_HEADS,
        num_layers: int = config.NUM_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model

        # Input Projection: num_features -> d_model
        self.input_projection = nn.Linear(num_features, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len=input_window + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Forecasting head: pool encoder outputs over time, then project to horizon.
        self.norm = nn.LayerNorm(d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_window, num_features)

        Returns
        -------
        torch.Tensor, shape (batch_size, forecast_horizon)
        """
        x = self.input_projection(x)                # (batch, input_window, d_model)
        x = self.positional_encoding(x)              # (batch, input_window, d_model)
        x = self.transformer_encoder(x)               # (batch, input_window, d_model)

        # Mean-pool across the time dimension to summarize the whole window,
        # then map the summary to the forecast horizon.
        pooled = x.mean(dim=1)                        # (batch, d_model)
        pooled = self.norm(pooled)
        out = self.forecast_head(pooled)               # (batch, forecast_horizon)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
