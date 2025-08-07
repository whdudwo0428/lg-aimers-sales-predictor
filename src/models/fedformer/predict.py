"""Inference script for the FedFormer wrapper.

This module loads a trained model checkpoint from ``checkpoint/fedformer``
and applies it to the test CSVs in the ``dataset/test`` directory.  The
resulting predictions are written to ``results/fedformer_submission.csv``
in the same format as ``dataset/sample_submission.csv``.

At present this implementation uses a trivial forecasting strategy:
predicted sales are set to zero for all items.  You should replace
the body of ``generate_predictions`` with logic that constructs
appropriate input sequences from the test data and runs the model
defined in ``model.py`` to produce forecasts.
"""

from __future__ import annotations

import os
import pandas as pd
import torch

from ...core import data_loader, feature_engineer, utils
from .model import build_model, ModelConfig


def generate_predictions(cfg: ModelConfig) -> pd.DataFrame:
    """Generate predictions for the test set.

    Currently returns a DataFrame of zeros indexed by ``store_item``.

    Args:
        cfg: Model configuration (unused in the trivial baseline).

    Returns:
        DataFrame with columns ``store_item`` and ``sales``.
    """
    # Load sample submission to obtain the store_item keys
    # Determine project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sample_path = os.path.join(project_root, "dataset", "sample_submission.csv")
    sample_df = data_loader.load_sample_submission(sample_path)
    key_col = "store_item" if "store_item" in sample_df.columns else "id"
    preds = sample_df[[key_col]].copy()
    preds["sales"] = 0.0
    preds = preds.rename(columns={key_col: "store_item"})
    return preds


def main() -> None:
    cfg = ModelConfig()
    utils.seed_everything(cfg.seed)
    # Load checkpoint – not used for baseline but left here for completeness
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ckpt_dir = os.path.join(project_root, "checkpoint", "fedformer")
    ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint {ckpt_path} not found.  Predictions will be zeros.")
    # Build model and load weights if available
    # In the trivial baseline we ignore the model and output zeros.
    preds = generate_predictions(cfg)
    # Convert to submission format
    sample_path = os.path.join(project_root, "dataset", "sample_submission.csv")
    sample_df = data_loader.load_sample_submission(sample_path)
    submission = utils.convert_to_submission_format(preds, sample_df)
    # Save submission
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "fedformer_submission.csv")
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()