"""
Evaluation module for the LG Aimers Gonjiam Resort demand forecasting challenge.

This script can be run as a standalone program to compute the Symmetric
Mean Absolute Percentage Error (SMAPE) for predictions produced by the
models in this repository.  It reads the ground truth values and
predicted values in wide format (similar to ``sample_submission.csv``)
and computes SMAPE per item, per store and overall.  The final score
reported mirrors the competition rules as closely as possible given
that the true store weights are unknown.  In lieu of those weights,
this implementation uses simple averaging across stores.

Usage
-----
To evaluate a submission file against a ground truth CSV::

    python evaluate.py --truth path/to/ground_truth.csv --pred path/to/submission.csv --output results/breakdown.csv

The ``ground_truth.csv`` should contain the actual sales values for
the forecast horizon.  The format must match the sample submission
file: the first column is a date identifier (e.g. ``TEST_00+1일``) and
subsequent columns correspond to the same set of items in the same
order.  The ``submission.csv`` should have an identical header and
row ordering, with the predicted sales values.

The script writes a breakdown CSV detailing the SMAPE per item and
per store.  The final overall SMAPE (simple mean of store averages)
is printed to the console.

Note
----
This evaluation tool is intended for local validation.  The official
leaderboard uses a proprietary weighting per store that is not
publicly disclosed.  The aggregated score computed here will
therefore differ from the leaderboard, but relative rankings should be
comparable when consistent data and models are used.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from typing import Dict, List

from models.common import compute_smape_per_item


def load_wide_csv(path: str) -> pd.DataFrame:
    """Load a wide format CSV into a DataFrame with string dates.

    Parameters
    ----------
    path : str
        Path to the CSV file.  The first column is assumed to be an
        identifier (e.g. date string) and the remaining columns are item names.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by the identifier column.
    """
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    return df


def evaluate_submission(truth_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a prediction DataFrame against the ground truth.

    Parameters
    ----------
    truth_df : pandas.DataFrame
        Ground truth with index of horizon identifiers and columns of items.
    pred_df : pandas.DataFrame
        Predictions with matching index and columns.

    Returns
    -------
    pandas.DataFrame
        Breakdown of SMAPE per item and per store.
    """
    # Ensure columns match
    if list(truth_df.columns) != list(pred_df.columns):
        raise ValueError("Columns of prediction do not match truth")
    # Determine item → store mapping
    item_to_store = {item: item.split("_")[0] for item in truth_df.columns}
    # Convert wide into per item arrays
    y_true: Dict[str, np.ndarray] = {}
    y_pred: Dict[str, np.ndarray] = {}
    for item in truth_df.columns:
        y_true[item] = truth_df[item].values.reshape(-1, 1)  # shape (n_samples, 1)
        y_pred[item] = pred_df[item].values.reshape(-1, 1)
    # Compute SMAPE per item (single horizon when evaluating final test)
    smape_records = []
    store_smapes: Dict[str, List[float]] = {}
    for item, true_arr in y_true.items():
        pred_arr = y_pred[item]
        smape_val = compute_smape_per_item(true_arr, pred_arr)
        store = item_to_store[item]
        smape_records.append({
            "store": store,
            "item": item,
            "smape_item": smape_val
        })
        store_smapes.setdefault(store, []).append(smape_val)
    # Compute store averages
    store_avg_records = []
    for store, smapes in store_smapes.items():
        store_avg_records.append({
            "store": store,
            "item": "(average)",
            "smape_item": float(np.mean(smapes))
        })
    records_df = pd.DataFrame(smape_records + store_avg_records)
    return records_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMAPE for LG Aimers submissions")
    parser.add_argument("--truth", type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction CSV")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save breakdown CSV")
    args = parser.parse_args()
    truth_df = load_wide_csv(args.truth)
    pred_df = load_wide_csv(args.pred)
    breakdown = evaluate_submission(truth_df, pred_df)
    # Compute overall SMAPE (simple average of store averages)
    store_avgs = breakdown[breakdown["item"] == "(average)"]["smape_item"].values
    overall_smape = float(np.mean(store_avgs)) if len(store_avgs) > 0 else 0.0
    print(f"Overall SMAPE (simple mean over stores): {overall_smape:.6f}")
    if args.output:
        breakdown.to_csv(args.output, index=False)
        print(f"Breakdown saved to {args.output}")


if __name__ == "__main__":
    main()
