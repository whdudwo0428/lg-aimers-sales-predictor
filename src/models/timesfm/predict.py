"""Prediction script for the TimesFM wrapper.

Outputs zero predictions for all items.  Replace with real model
inference as required.
"""

from __future__ import annotations

import os
import pandas as pd

from ...core import data_loader, utils
from .model import ModelConfig


def generate_predictions(cfg: ModelConfig) -> pd.DataFrame:
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ckpt_dir = os.path.join(project_root, "checkpoint", "timesfm")
    ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint {ckpt_path} not found.  Predictions will be zeros.")
    preds = generate_predictions(cfg)
    sample_path = os.path.join(project_root, "dataset", "sample_submission.csv")
    sample_df = data_loader.load_sample_submission(sample_path)
    submission = utils.convert_to_submission_format(preds, sample_df)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "timesfm_submission.csv")
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()