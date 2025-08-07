"""PatchTST model configuration dataclass.

For the placeholder implementation this simply inherits from the
base configuration defined in ``core.utils.ModelConfigBase``.  Add
any PatchTST‑specific hyperparameters here when replacing the dummy
model with the true implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from ...core.utils import ModelConfigBase


@dataclass
class PatchTSTConfig(ModelConfigBase):
    # Placeholder for additional PatchTST hyperparameters
    patch_size: int = 16