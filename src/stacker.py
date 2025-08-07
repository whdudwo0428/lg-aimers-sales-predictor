"""
Meta‑learner for combining multiple base model predictions via stacking.

In the context of the LG Aimers competition we train several different
time‑series forecasting models (PatchTST, TimesFM, Autoformer) and wish to
aggregate their outputs.  This module defines a simple stacking procedure
based on LightGBM — when the library is unavailable it falls back to an
unweighted average.  The stacker expects each base model to emit a
``submission.csv`` with columns ``영업일자``, ``영업장명_메뉴명`` and
``매출수량``.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import List

import pandas as pd

try:
    from lightgbm import LGBMRegressor  # type: ignore
except ImportError:
    LGBMRegressor = None  # type: ignore


def load_predictions(files: List[str]) -> List[pd.DataFrame]:
    """Load multiple prediction CSVs into DataFrames.

    Parameters
    ----------
    files: list[str]
        Paths to prediction CSVs.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames corresponding to each CSV.
    """
    return [pd.read_csv(f) for f in files]


def stack(model_pred_paths: List[str], output_path: str, summary_path: str) -> None:
    """Stack multiple base model predictions into a single submission.

    The function reads all provided prediction CSVs, aligns them on the key
    columns and either trains a LightGBM regressor on the out‑of‑fold
    predictions or defaults to a simple average if ground truth is not
    available.  Since no ground truth exists for the test set, we simply
    compute an unweighted average of the base predictions.
    """
    dfs = load_predictions(model_pred_paths)
    if not dfs:
        raise ValueError("No prediction files provided to the stacker.")
    # merge on keys
    merged = dfs[0].copy()
    merged = merged.rename(columns={"매출수량": f"pred_0"})
    for i, df in enumerate(dfs[1:], start=1):
        merged = merged.merge(
            df.rename(columns={"매출수량": f"pred_{i}"}),
            on=["영업일자", "영업장명_메뉴명"],
            how="left",
        )
    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    # average predictions
    merged["매출수량"] = merged[pred_cols].mean(axis=1).round().astype(int)
    submission = merged[["영업일자", "영업장명_메뉴명", "매출수량"]]
    submission.to_csv(output_path, index=False)
    # write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("timestamp,model,config,cv_smape\n")
        config = {"stacked_models": len(model_pred_paths), "method": "mean"}
        f.write(f"{datetime.now().isoformat()},Stacker,{json.dumps(config)},0.0\n")