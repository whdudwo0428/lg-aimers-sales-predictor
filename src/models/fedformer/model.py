"""Wrapper for the FedFormer architecture used in the LG Aimers pipeline.

For the purposes of this example project, we define a simple linear
model that serves as a placeholder.  In a full implementation you
should dynamically import the upstream FedFormer implementation
present in the ``models/FEDformer`` directory and construct it here.

This wrapper exposes a single class ``ForecastModel`` with a PyTorch
interface: it accepts an input tensor of shape ``(batch, seq_len,
num_features)`` and returns a prediction tensor of shape ``(batch,
forecast_horizon)``.  You can replace the internals of this class
with any architecture of your choosing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass


class ForecastModel(nn.Module):
    """Very simple forecasting model used as a placeholder.

    The model computes the mean over the input sequence and applies a
    linear transformation to predict the next ``forecast_horizon``
    values.  While trivial, this demonstrates the expected input and
    output shapes.
    """

    def __init__(self, input_dim: int, forecast_horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch, seq_len, input_dim)
        # Reduce over sequence dimension
        x_mean = x.mean(dim=1)
        return self.linear(x_mean)


@dataclass
class ModelConfig:
    """FedFormer wrapper configuration.

    In a real setup you would include the full set of hyperparameters
    required by the upstream FedFormer implementation here.  This
    placeholder keeps only those parameters needed for the dummy
    model.
    """

    seed: int = 42
    input_length: int = 28
    forecast_horizon: int = 7
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 10


def build_model(input_dim: int, config: ModelConfig) -> ForecastModel:
    """Instantiate the forecast model.

    Args:
        input_dim: Number of features per timestep.
        config: Model hyperparameters.

    Returns:
        A PyTorch ``nn.Module``.
    """
    return ForecastModel(input_dim=input_dim, forecast_horizon=config.forecast_horizon)