"""Utilities for data processing and evaluation.

The ``core`` package contains reusable components that are agnostic
to any specific model implementation.  These include data loading
functions, simple feature engineering helpers, PyTorch‑Lightning
datamodules for sliding‑window sequence generation, common loss
functions and evaluation metrics, holiday definitions and misc
utilities.
"""

__all__ = [
    "data_loader",
    "data_module",
    "feature_engineer",
    "loss",
    "evaluation",
    "holidays",
    "utils",
]