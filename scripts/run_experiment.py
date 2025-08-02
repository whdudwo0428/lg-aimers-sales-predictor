"""
Run training and inference for a specified forecasting model.

This script serves as the entry point for conducting experiments.  It
handles loading the dataset, constructing sliding windows, splitting
into training and validation sets, fitting the chosen model, evaluating
on the validation set and generating predictions for the test set.

Usage example::

    python scripts/run_experiment.py \
        --model tft \
        --params iterations=300,learning_rate=0.03 \
        --lookback 28 --horizon 7 --val_ratio 0.2 \
        --output_dir results

The predictions will be saved as a CSV file under ``output_dir`` with
a filename reflecting the model name and key hyperparameters.  A
summary row containing validation SMAPE will be appended to
``experiment_summary.csv`` within the same directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Append parent directory to sys.path so that `models` can be imported when
# executing this script directly.  Without this adjustment Python will
# fail to locate the `models` package because it is not installed as a
# site‑package.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.common import Config, DataLoader, evaluate_predictions, compute_smape_per_item
from models.base import build_model
from models.deepar import DeepARModel


def parse_params(param_str: str) -> Dict[str, any]:
    """Parse a comma separated list of key=value pairs into a dict."""
    params: Dict[str, any] = {}
    if not param_str:
        return params
    for pair in param_str.split(","):
        if not pair:
            continue
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Try to convert to numeric or boolean
        if val.lower() in {"true", "false"}:
            params[key] = val.lower() == "true"
        else:
            try:
                if "." in val:
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = val
    return params


def main():
    parser = argparse.ArgumentParser(description="Run forecasting experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name (tft, nbeats, dlinear, autoformer, fedformer, patchtst, deepar, gbt, sliding_transformer, hybrid)")
    parser.add_argument("--params", type=str, default="", help="Model hyperparameters as comma separated key=value pairs")
    parser.add_argument("--lookback", type=int, default=28, help="Number of past days to use as input")
    parser.add_argument("--horizon", type=int, default=7, help="Number of future days to forecast")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of windows used for validation")
    parser.add_argument("--train_path", type=str, default="dataset/train/train.csv", help="Path to training CSV")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="Directory containing test CSVs")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Parse hyperparameters
    model_params = parse_params(args.params)
    # Initialise data loader
    config = Config(lookback=args.lookback, horizon=args.horizon, test_dir=args.test_dir)
    loader = DataLoader(train_path=args.train_path, test_dir=args.test_dir, config=config)
    # Build model
    model_name = args.model.lower()
    # Some models require special treatment
    if model_name == "deepar":
        # Prepare per item series for ARIMA fitting
        train_df = loader.train_df.copy()
        # pivot to ensure continuous index across items
        pivot = train_df.pivot(index="영업일자", columns="영업장명_메뉴명", values="매출수량")
        # fill missing dates with zeros
        pivot = pivot.fillna(0)
        series_dict = {col: pivot[col] for col in pivot.columns}
        model = DeepARModel(horizon=args.horizon, **model_params)
        model.fit_series(series_dict)
        # For validation we will evaluate using sliding windows
        X_train, y_train, X_val, y_val = loader.get_train_val_split(val_ratio=args.val_ratio)
        # Derive item names for each row in validation based on item_id encoded in features
        # In DataLoader, item_id is stored after past_vals in feature vector: position lookback
        item_ids_encoded = loader.item_encoder
        # compute mapping from encoded id to item name
        id_to_item = {i: item for i, item in enumerate(item_ids_encoded.classes_)}
        # Extract encoded item ids from X_val
        # item_id is at index lookback (0‑based) of each row
        encoded_ids = X_val[:, args.lookback].astype(int)
        item_names = [id_to_item.get(int(i), "") for i in encoded_ids]
        # Predictions
        y_pred = model.predict(X_val, item_names)
        # Evaluate smape per item
        # Build mapping y_true dict and y_pred dict
        y_true_dict: Dict[str, np.ndarray] = {}
        y_pred_dict: Dict[str, np.ndarray] = {}
        for idx, item in enumerate(item_names):
            y_true_dict.setdefault(item, []).append(y_val[idx])
            y_pred_dict.setdefault(item, []).append(y_pred[idx])
        # Convert lists to arrays
        for k in y_true_dict.keys():
            y_true_dict[k] = np.vstack(y_true_dict[k])
            y_pred_dict[k] = np.vstack(y_pred_dict[k])
        # Map item to store
        item_to_store = {item: item.split("_")[0] for item in y_true_dict.keys()}
        breakdown = evaluate_predictions(y_true_dict, y_pred_dict, item_to_store)
        # Compute overall smape
        store_avgs = breakdown[breakdown["item"] == "(average)"]["smape_item"].values
        val_smape = float(np.mean(store_avgs)) if len(store_avgs) > 0 else 0.0
        print(f"Validation SMAPE: {val_smape:.6f}")
    else:
        model = build_model(model_name, horizon=args.horizon, **model_params)
        # Create training and validation sets
        X_train, y_train, X_val, y_val = loader.get_train_val_split(val_ratio=args.val_ratio)
        if X_train.shape[0] == 0:
            print("No training samples generated. Check if the training data has enough history.")
            return
        # Fit model
        model.fit(X_train, y_train)
        # Validation predictions
        y_pred_val = model.predict(X_val)
        # Evaluate SMAPE per item
        # Build mapping item->true/pred arrays
        # Extract item names from X_val using item_id index (lookback index)
        lookback = args.lookback
        id_to_item = {i: item for i, item in enumerate(loader.item_encoder.classes_)}
        encoded_ids = X_val[:, lookback].astype(int)
        y_true_dict: Dict[str, List[np.ndarray]] = {}
        y_pred_dict: Dict[str, List[np.ndarray]] = {}
        for idx, enc in enumerate(encoded_ids):
            item = id_to_item.get(int(enc), "")
            y_true_dict.setdefault(item, []).append(y_val[idx])
            y_pred_dict.setdefault(item, []).append(y_pred_val[idx])
        for k in y_true_dict.keys():
            y_true_dict[k] = np.vstack(y_true_dict[k])
            y_pred_dict[k] = np.vstack(y_pred_dict[k])
        item_to_store = {item: item.split("_")[0] for item in y_true_dict.keys()}
        breakdown = evaluate_predictions(y_true_dict, y_pred_dict, item_to_store)
        store_avgs = breakdown[breakdown["item"] == "(average)"]["smape_item"].values
        val_smape = float(np.mean(store_avgs)) if len(store_avgs) > 0 else 0.0
        print(f"Validation SMAPE: {val_smape:.6f}")
    # Generate predictions for test set
    if os.path.exists(args.test_dir):
        X_test_list = loader.get_test_features()
        if X_test_list:
            # Determine the path to the submission template.  The template is expected to live
            # alongside the training CSV (e.g. ``dataset/train/sample_submission.csv``).  If
            # it isn't found there, fall back to the parent directory (``dataset/``) so that
            # users who keep the template in the dataset root will still be supported.
            sample_sub_path = os.path.join(os.path.dirname(args.train_path), "sample_submission.csv")
            if not os.path.exists(sample_sub_path):
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(args.train_path), os.pardir))
                fallback = os.path.join(parent_dir, "sample_submission.csv")
                if os.path.exists(fallback):
                    sample_sub_path = fallback
            sample_sub = pd.read_csv(sample_sub_path)
            item_order = list(sample_sub.columns[1:])
            # Prepare result DataFrame with same shape as sample submission
            submission_rows: List[pd.DataFrame] = []
            for idx_file, X_test in enumerate(X_test_list):
                if model_name == "deepar":
                    # Determine item names for this test file
                    item_names = item_order
                    preds = model.predict(X_test, item_names)
                else:
                    preds = model.predict(X_test)
                # Create DataFrame with horizon rows for this test file
                rows = {}
                for h in range(args.horizon):
                    horizon_label = f"TEST_{str(idx_file).zfill(2)}+{h+1}일"
                    rows[horizon_label] = preds[:, h]
                df = pd.DataFrame(rows).T
                df.columns = item_order
                submission_rows.append(df)
            submission_df = pd.concat(submission_rows, axis=0)
            # Reset index to ensure first column is date label
            submission_df.reset_index(inplace=True)
            submission_df.rename(columns={"index": "영업일자"}, inplace=True)
            # Derive output filename
            param_repr = "_".join([f"{k}{v}" for k, v in model_params.items() if isinstance(v, (int, float, str))][:3])
            if not param_repr:
                param_repr = "default"
            out_name = f"submission_{args.model}_{param_repr}.csv"
            out_path = os.path.join(args.output_dir, out_name)
            submission_df.to_csv(out_path, index=False)
            print(f"Submission saved to {out_path}")
            # Append summary entry
            summary_path = os.path.join(args.output_dir, "experiment_summary.csv")
            summary_row = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "params": json.dumps(model_params),
                "val_smape": val_smape,
                "submission_file": out_name,
            }
            summary_df = pd.DataFrame([summary_row])
            if os.path.exists(summary_path):
                existing = pd.read_csv(summary_path)
                summary_df = pd.concat([existing, summary_df], ignore_index=True)
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary updated at {summary_path}")
        else:
            print("No test files found; skipping submission generation.")


if __name__ == "__main__":
    main()
