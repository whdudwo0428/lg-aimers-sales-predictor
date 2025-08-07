"""TimesFM model wrapper placeholder.

See ``src/models/fedformer/model.py`` for a description of the
placeholder architecture.  Replace ``ForecastModel`` with the
TimesFM implementation from the official repository when ready.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass


class ForecastModel(nn.Module):
    def __init__(self, input_dim: int, forecast_horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=1))


@dataclass
class ModelConfig:
    seed: int = 42
    input_length: int = 28
    forecast_horizon: int = 7
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 10


def build_model(input_dim: int, config: ModelConfig) -> ForecastModel:
    return ForecastModel(input_dim=input_dim, forecast_horizon=config.forecast_horizon)