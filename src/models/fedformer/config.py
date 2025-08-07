"""Hyperparameter configuration for the FedFormer wrapper.

This module defines a ``FedFormerConfig`` dataclass which extends
``core.utils.ModelConfigBase`` with any additional parameters needed
by your model.  The provided defaults are conservative and should
serve as a reasonable starting point for experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Import the base configuration from the core utilities.  The three
# leading dots navigate up from src/models/fedformer to src/core.
from ...core.utils import ModelConfigBase


@dataclass
class FedFormerConfig(ModelConfigBase):
    """Configuration specific to the FedFormer model.

    Extend the base config with any FedFormer‑specific parameters
    such as model dimension, number of heads/layers etc.  For the
    placeholder implementation these parameters are unused.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    # Add additional parameters as required by the real FedFormer implementation