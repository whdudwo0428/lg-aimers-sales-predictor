"""Top-level package for LG Aimers forecasting pipeline.

This package is structured to separate generic, model‑agnostic
components under ``core`` from individual model wrappers under
``models``.  Training and prediction entrypoints live in the root of
``src`` and simply delegate into the appropriate subpackage.  See
``README.md`` in this directory for usage instructions.
"""

__all__ = []