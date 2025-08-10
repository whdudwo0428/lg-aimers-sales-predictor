"""
Abstract base class for forecasting models.

All models should inherit from :class:`BaseModel` and implement a
``forward`` method.  The forward method should accept a batch
dictionary and return a tensor of shape ``(B, horizon, N)`` where
``B`` is the batch size, ``horizon`` is the forecast length and
``N`` is the number of series.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import torch.nn as nn  # type: ignore
    import torch  # type: ignore
except ImportError:
    # torch is optional.  Provide a dummy nn.Module to allow the class
    # definition to succeed even when torch is absent.
    nn = None  # type: ignore
    torch = None  # type: ignore


if nn is not None:
    class BaseModel(nn.Module, ABC):
        """Abstract base class for all forecasting models."""

        @abstractmethod
        def forward(self, batch: dict) -> torch.Tensor:
            """Forward pass for the model.

            Parameters
            ----------
            batch : dict
                Contains inputs to the model.  At a minimum this should
                include the key ``x_enc`` which is a tensor of shape
                ``(B, seq_len, N)``.  Additional keys such as
                ``x_mark_enc`` and ``y_mark_dec`` may also be provided.

            Returns
            -------
            torch.Tensor
                Predictions with shape ``(B, horizon, N)``.
            """
            raise NotImplementedError
else:
    class BaseModel(ABC):
        """Fallback base class when PyTorch is unavailable.

        Models cannot be instantiated without PyTorch; this class
        provides the interface for type checking only.  Attempting to
        instantiate or call methods on subclasses will raise
        ImportError.
        """

        @abstractmethod
        def forward(self, batch: dict) -> Any:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use models derived from BaseModel. "
                "Please install torch and pytorch_lightning."
            )


__all__ = ["BaseModel"]