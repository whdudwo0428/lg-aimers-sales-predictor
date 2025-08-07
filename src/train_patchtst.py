"""Entrypoint for training the PatchTST model.

Delegates to ``src.models.patchtst.train.main`` so that you can run

    python -m src.train_patchtst

from the project root.
"""

from __future__ import annotations

from .models.patchtst.train import main


if __name__ == "__main__":
    main()