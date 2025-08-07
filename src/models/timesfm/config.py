"""TimesFM model configuration dataclass.

Extends the base configuration with any parameters specific to the
TimesFM architecture.  For now this remains identical to the base
configuration since the placeholder model is a simple linear
regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from ...core.utils import ModelConfigBase


@dataclass
class TimesFMConfig(ModelConfigBase):
    # Placeholder for any TimesFM specific parameters
    pass