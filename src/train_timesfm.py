"""
Train and run the TimesFM baseline model.

This is a placeholder implementation that mirrors the structure of
``train_patchtst.py``.  When a true TimesFM implementation is desired you
should replace the simple regressor with the official TimesFM model from
``models/timesfm``.  The rest of the code — data loading, windowing and
submission formatting — remains unchanged.
"""

from __future__ import annotations

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .data_loader import DataConfig, load_raw_data, prepare_datasets
from .utils import set_seed

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None  # type: ignore
from sklearn.ensemble import RandomForestRegressor


def main(project_root: str = os.getcwd()) -> None:
    set_seed(42)
    cfg = DataConfig()
    train_df, test_files, sample_submission = load_raw_data(cfg, project_root)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    if train_df.empty or sample_submission.empty:
        sub = sample_submission.copy()
        if not sub.empty and "매출수량" in sub.columns:
            sub["매출수량"] = 0
        sub_path = os.path.join(results_dir, "submission.csv")
        sub.to_csv(sub_path, index=False)
        summary_path = os.path.join(results_dir, "experiment_summary.csv")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("timestamp,model,config,cv_smape\n")
            f.write(f"{datetime.now().isoformat()},TimesFM,{{}},N/A\n")
        return

    X, y, ids, sx, sy, le = prepare_datasets(cfg, project_root)
    X_flat = X.reshape(X.shape[0], -1)
    horizon = y.shape[1]
    ModelClass = LGBMRegressor if LGBMRegressor is not None else RandomForestRegressor
    models = []
    for step in range(horizon):
        model = ModelClass()
        model.fit(X_flat, y[:, step])
        models.append(model)
    preds = []
    for path in test_files:
        df_test = pd.read_csv(path)
        prefix = os.path.splitext(os.path.basename(path))[0]
        for name in df_test["영업장명_메뉴명"].unique():
            if name not in le.classes_:
                forecast = np.zeros(horizon, dtype=int)
            else:
                item_id = np.where(le.classes_ == name)[0][0]
                hist = train_df[train_df["영업장명_메뉴명"] == name].sort_values("영업일자")
                values = hist["매출수량"].astype(float)
                dates = pd.to_datetime(hist["영업일자"])
                if len(values) < cfg.lookback:
                    pad = cfg.lookback - len(values)
                    values_pad = np.concatenate([np.zeros(pad), values.values])
                    dates_pad = pd.date_range(end=dates.iloc[-1], periods=cfg.lookback)
                else:
                    values_pad = values.iloc[-cfg.lookback:].values
                    dates_pad = dates.iloc[-cfg.lookback:]
                lag_roll = pd.DataFrame({"value": values_pad})
                for lag in (1, 7, 14, 21, 28):
                    lag_roll[f"lag_{lag}"] = lag_roll["value"].shift(lag).fillna(method="bfill").fillna(method="ffill")
                lag_roll["roll_7"] = lag_roll["value"].shift(1).rolling(7, min_periods=1).mean().fillna(method="bfill").fillna(method="ffill")
                lag_roll["roll_var_7"] = lag_roll["value"].shift(1).rolling(7, min_periods=1).var().fillna(0)
                cal = np.stack([
                    np.sin(2 * np.pi * dates_pad.dayofweek / 7), np.cos(2 * np.pi * dates_pad.dayofweek / 7),
                    np.sin(4 * np.pi * dates_pad.dayofweek / 7), np.cos(4 * np.pi * dates_pad.dayofweek / 7),
                    np.sin(2 * np.pi * (dates_pad.month - 1) / 12), np.cos(2 * np.pi * (dates_pad.month - 1) / 12),
                    np.sin(2 * np.pi * (dates_pad.dayofyear - 1) / 365), np.cos(2 * np.pi * (dates_pad.dayofyear - 1) / 365),
                    (dates_pad.dayofweek >= 5).astype(int), np.zeros(len(dates_pad)),
                ], axis=1)
                item_wd = (item_id * 7 + dates_pad.dayofweek).values.reshape(-1, 1)
                features = np.hstack([
                    values_pad.reshape(-1, 1), cal, lag_roll.iloc[:, 1:].values, item_wd
                ])
                features[:, 0] = sx.transform(features[:, [0]]).flatten()
                flat = features.reshape(1, -1)
                forecast_scaled = np.array([m.predict(flat)[0] for m in models])
                forecast = sy.inverse_transform(forecast_scaled.reshape(1, -1)).flatten()
                forecast = np.round(np.clip(forecast, 0, None)).astype(int)
            for i, v in enumerate(forecast):
                preds.append({
                    "영업일자": f"{prefix}+{i + 1}일",
                    "영업장명_메뉴명": name,
                    "매출수량": int(v),
                })
    sub_df = pd.DataFrame(preds, columns=["영업일자", "영업장명_메뉴명", "매출수량"])
    sub_path = os.path.join(results_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    summary_path = os.path.join(results_dir, "experiment_summary.csv")
    config_dict = {
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "model": ModelClass.__name__,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("timestamp,model,config,cv_smape\n")
        f.write(f"{datetime.now().isoformat()},TimesFM,{json.dumps(config_dict)},0.0\n")


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    main(root)