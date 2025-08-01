"""
Inference script for the Gonjiam resort demand forecasting challenge.

This module loads the trained model and baseline statistics, prepares
features for the forecasting horizon (7 days) for each test file, and
generates predicted sales quantities.  It outputs a dictionary mapping
`TEST_XX` identifiers to prediction DataFrames, which can then be
converted into the submission format.
"""
import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

try:
    from .util import add_date_features
except ImportError:
    from util import add_date_features


def load_model(model_path: Path):
    """Load the saved model and metadata from disk."""
    data = joblib.load(model_path)
    model = data['model']
    categories = data['menu_categories']
    return model, categories


def infer_predictions(model, categories: List[str], baseline_means: pd.Series,
                      test_dir: Path) -> Dict[str, pd.DataFrame]:
    """Generate predictions for each TEST file.

    Parameters
    ----------
    model : trained model
        The loaded regression model used to generate predictions.
    categories : list of str
        Ordered list of menu categories used during training.
    baseline_means : pd.Series
        Precomputed baseline means indexed by menu name.
    test_dir : Path
        Directory containing the TEST_XX.csv files.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from test identifier (e.g. 'TEST_00') to a DataFrame of
        predicted values with index representing the next 7 days.
    """
    predictions: Dict[str, pd.DataFrame] = {}
    categories_map = {cat: idx for idx, cat in enumerate(categories)}
    # Determine global median baseline in case a menu is missing from baseline_means
    global_baseline = baseline_means.mean() if len(baseline_means) > 0 else 0.0

    for test_file in sorted(test_dir.glob('TEST_*.csv')):
        test_name = test_file.stem  # e.g. TEST_00
        df_test = pd.read_csv(test_file)
        df_test['영업일자'] = pd.to_datetime(df_test['영업일자'])
        # Unique menus in this test file
        menus_in_test = df_test['영업장명_메뉴명'].unique().tolist()
        # Determine the last date in test for each menu
        last_date = df_test['영업일자'].max()
        # Prepare a DataFrame to collect predictions (rows: next 7 days, columns: menus)
        rows = [f"{test_name}+{i}일" for i in range(1, 8)]
        pred_df = pd.DataFrame(index=rows, columns=menus_in_test, dtype=float)

        for menu in menus_in_test:
            # Build 7-day forecast dates
            dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
            # Build feature DataFrame for these dates
            tmp = pd.DataFrame({'영업일자': dates})
            tmp['영업장명_메뉴명'] = menu
            tmp = add_date_features(tmp, date_col='영업일자')
            # Encode menu code using training categories
            if menu in categories_map:
                tmp['menu_code'] = categories_map[menu]
            else:
                # Assign an out-of-range code for unseen menu
                tmp['menu_code'] = -1
            feature_cols = [
                'menu_code', 'year', 'month', 'day', 'dayofweek', 'weekofyear',
                'is_weekend', 'is_holiday', 'is_holiday_eve', 'is_holiday_after',
                'quarter', 'dayofyear', 'season', 'ski_season'
            ]
            X_feat = tmp[feature_cols].values
            # Predict using model
            try:
                preds = model.predict(X_feat)
            except Exception:
                preds = np.full(len(dates), global_baseline)
            # Replace negative predictions with zero
            preds = np.clip(preds, a_min=0, a_max=None)
            # If model fails or menu unseen, fallback to baseline mean
            if menu not in categories_map or np.any(np.isnan(preds)):
                baseline_val = baseline_means.get(menu, global_baseline)
                preds = np.full(len(dates), baseline_val)
            # Round to nearest integer (sales are counts)
            pred_df[menu] = np.round(preds).astype(int)

        predictions[test_name] = pred_df
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test datasets')
    parser.add_argument('--model-path', type=str, default='open/models/saved_models/rf_model.pkl', help='Path to saved model file')
    parser.add_argument('--baseline-path', type=str, default='open/models/saved_models/baseline_means.csv', help='Path to baseline means CSV')
    parser.add_argument('--test-dir', type=str, default='open/test', help='Directory containing test CSV files')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    baseline_path = Path(args.baseline_path)
    test_dir = Path(args.test_dir)

    model, categories = load_model(model_path)
    baseline_means = pd.read_csv(baseline_path, index_col=0).squeeze("columns")
    preds = infer_predictions(model, categories, baseline_means, test_dir)
    # For demonstration, save individual prediction DataFrames
    for name, df in preds.items():
        df.to_csv(f'open/results/{name}_preds.csv')
        print(f"Saved predictions for {name} to open/results/{name}_preds.csv")

if __name__ == '__main__':
    main()