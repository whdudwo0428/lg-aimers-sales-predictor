"""Convenience entrypoint for FedFormer training.

This script simply delegates to the ``main`` function in
``src.models.fedformer.train``.  It is provided so that you can run

::

    python -m src.train_fedformer

from the project root without worrying about the internal module
structure.  See ``src/models/fedformer/train.py`` for the
implementation details.
"""

from __future__ import annotations

from .models.fedformer.train import main


if __name__ == "__main__":
    main()