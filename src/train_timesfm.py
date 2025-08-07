"""Entrypoint for training the TimesFM model."""

from __future__ import annotations

from .models.timesfm.train import main


if __name__ == "__main__":
    main()