# src/predict_fedformer.py
from __future__ import annotations

import os, re, glob, argparse, datetime
import numpy as np
import pandas as pd
import torch

from .config import Config
from .core.data_module import TimeSeriesDataModule
from .core.feature_engineer import FeatureEngineer, add_date_features
from .models.model_fedformer import FedformerModel
from .core.lightning_module import LitModel

def _find_ckpt(cfg: Config, explicit: str | None) -> str:
    if explicit:
        return explicit
    patt = os.path.join(cfg.CHECKPOINT_DIR, "fedformer_*.ckpt")
    cks = sorted(glob.glob(patt), key=os.path.getmtime, reverse=True)
    if not cks:
        raise FileNotFoundError(f"No checkpoint found under {cfg.CHECKPOINT_DIR}")
    return cks[0]

def _param_tag(cfg: Config) -> str:
    f = cfg.FEDformer
    return f"d{f.D_MODEL}_L{f.E_LAYERS}_seq{cfg.SEQ_LEN}_h{cfg.HORIZON}_bs{cfg.BATCH_SIZE}_lr{cfg.LR}"

def _to_submission_format(pred_df: pd.DataFrame, sample_submission_path: str) -> pd.DataFrame:
    # LSTM 예시와 동일한 변환 로직
    sample_submission = pd.read_csv(sample_submission_path)
    pred_dict = dict(zip(zip(pred_df["영업일자"], pred_df["영업장명_메뉴명"]), pred_df["매출수량"]))
    final_df = sample_submission.copy()
    for row_idx in final_df.index:
        date = final_df.loc[row_idx, "영업일자"]
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)
    return final_df  # 【참고】LSTM 변환 로직【turn16file1†lstm.py†L29-L39】

@torch.no_grad()
def main():
    cfg = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path (optional)")
    ap.add_argument("--outdir", type=str, default=cfg.RESULTS_DIR)
    args = ap.parse_args()

    ckpt_path = _find_ckpt(cfg, args.ckpt)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # --- Data schema from training ---
    fe = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(
        file_path=cfg.TRAIN_FILE,
        sequence_length=cfg.SEQ_LEN,
        forecast_horizon=cfg.HORIZON,
        label_len=cfg.LABEL_LEN,
        batch_size=cfg.BATCH_SIZE,
        feature_engineer=fe,
        num_workers=0,
    )
    dm.prepare_data()  # restore target columns & time feature names

    item_names = list(dm.target_columns)
    time_cols = list(dm.time_feature_columns)
    n_items = len(item_names)

    # --- Model & checkpoint ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FedformerModel.from_config(cfg, input_dim=n_items).to(device)
    lit = LitModel(model=model, cfg=cfg, item_names=item_names).to(device)

    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = lit.load_state_dict(state["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] missing={missing}, unexpected={unexpected}")

    model.eval()

    # --- Predict per TEST_xx.csv ---
    results_long = []
    test_files = sorted(glob.glob(os.path.join(cfg.TEST_DIR, "TEST_*.csv")))
    for path in test_files:
        test_df = pd.read_csv(path)
        # into training schema (same columns & time features)
        prepared = dm.preprocess_inference_data(test_df)

        if len(prepared) < cfg.SEQ_LEN:
            print(f"[skip] {os.path.basename(path)} has only {len(prepared)} rows (< SEQ_LEN).")
            continue

        # encoder inputs
        x_all = prepared[item_names].astype(np.float32)
        x_enc_np = x_all.tail(cfg.SEQ_LEN).to_numpy()                    # (seq, N)
        x_mark_enc_np = prepared[time_cols].tail(cfg.SEQ_LEN).astype(np.float32).to_numpy()  # (seq, T)

        # decoder time marks: last LABEL_LEN days + next HORIZON days
        hist_idx = prepared.index[-cfg.LABEL_LEN:]
        fut_idx = pd.date_range(prepared.index[-1] + pd.Timedelta(days=1),
                                periods=cfg.HORIZON, freq="D")
        # hist part from prepared, future part via add_date_features
        hist_marks = prepared.loc[hist_idx, time_cols].astype(np.float32)
        fut_marks = add_date_features(pd.DataFrame({"영업일자": fut_idx}), "영업일자")
        fut_marks.set_index("영업일자", inplace=True)
        fut_marks = fut_marks[time_cols].astype(np.float32)
        y_mark_dec_np = pd.concat([hist_marks, fut_marks], axis=0).to_numpy(dtype=np.float32)  # (label_len+h, T)

        # Pack batch dict expected by model wrapper
        batch = {
            "x_enc": torch.from_numpy(x_enc_np).unsqueeze(0).to(device).float(),
            "x_mark_enc": torch.from_numpy(x_mark_enc_np).unsqueeze(0).to(device).float(),
            "y_mark_dec": torch.from_numpy(y_mark_dec_np).unsqueeze(0).to(device).float(),
        }

        yhat = model(batch)  # (1, H, N)  — wrapper forwards to original
        yhat = yhat.squeeze(0).cpu().numpy()  # (H, N)

        # clip & round like baseline
        yhat = np.clip(yhat, 0, None)
        yhat = np.rint(yhat).astype(int)

        test_prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(cfg.HORIZON)]  # 【참고】LSTM과 동일 규칙【turn16file2†lstm.py†L40-L46】

        for step, d in enumerate(pred_dates):
            for j, col in enumerate(item_names):
                results_long.append({"영업일자": d, "영업장명_메뉴명": col, "매출수량": int(yhat[step, j])})

    pred_long_df = pd.DataFrame(results_long)
    if pred_long_df.empty:
        raise RuntimeError("No predictions produced. Check test files and sequence lengths.")

    # --- Merge to submission & save ---
    submission = _to_submission_format(pred_long_df, cfg.SAMPLE_SUBMISSION)
    param_tag = _param_tag(cfg)
    out_name = f"submission_fedformer_{param_tag}.csv"
    out_path = os.path.join(cfg.RESULTS_DIR, out_name)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[ok] submission saved → {out_path}")

    # optional: experiment_summary (same as LSTM)
    summary_path = os.path.join(cfg.RESULTS_DIR, "experiment_summary.csv")
    row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "fedformer", param_tag, ""]
    import csv
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["타임스탬프", "모델명", "파라미터", "점수"])
            csv.writer(f).writerow(row)
    else:
        with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)

if __name__ == "__main__":
    main()
