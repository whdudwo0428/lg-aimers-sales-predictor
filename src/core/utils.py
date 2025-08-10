"""
Miscellaneous utility functions.

Currently this module provides a convenience function to set random
seeds across Python, Numpy and PyTorch.  Additional helpers can be
added here as needed.
"""

from __future__ import annotations

import os
import random
import numpy as np
try:
    import torch  # type: ignore
except ImportError:
    # torch is optional; if not installed the functions depending on
    # torch will silently skip PyTorch seeding.  This allows the
    # utilities module to be imported in environments without PyTorch.
    torch = None  # type: ignore


def seed_everything(seed: int = 42) -> None:
    """Seed all relevant random number generators.

    This function seeds Python’s built‑in ``random`` module, NumPy and
    optionally PyTorch if it is available.  Setting the seed
    reproducibly controls sources of randomness across the entire
    pipeline.  If PyTorch is not installed the PyTorch‑specific
    portions of this function are skipped.

    Parameters
    ----------
    seed : int, default 42
        The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        # Attempt to seed CUDA if available
        if hasattr(torch.cuda, "manual_seed"):
            torch.cuda.manual_seed(seed)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)
        # Configure CuDNN for determinism
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


__all__ = ["seed_everything"]