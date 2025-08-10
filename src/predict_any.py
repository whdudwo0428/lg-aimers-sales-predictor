from __future__ import annotations
import os, re, glob, argparse, datetime
import numpy as np, pandas as pd, torch
from .config import Config
from .core.data_module import TimeSeriesDataModule
from .core.feature_engineer import FeatureEngineer, add_date_features
from .models.factory import build_model
from .core.lightning_module import LitModel  # ckpt 호환 로딩

def _find_ckpt(cfg: Config, model_name: str, explicit: str | None) -> str:
    if explicit: return explicit
    patt = os.path.join(cfg.CHECKPOINT_DIR, f"{model_name}_*.ckpt")
    cks = sorted(glob.glob(patt), key=os.path.getmtime, reverse=True)
    if not cks:
        raise FileNotFoundError(f"No ckpt for {model_name} under {cfg.CHECKPOINT_DIR}")
    return cks[0]

def _param_tag(cfg: Config) -> str:
    f = cfg.FEDformer
    return f"d{f.D_MODEL}_L{f.E_LAYERS}_seq{cfg.SEQ_LEN}_h{cfg.HORIZON}_bs{cfg.BATCH_SIZE}_lr{cfg.LR}"

def _to_submission(pred_df: pd.DataFrame, sample_path: str) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    pred_dict = dict(zip(zip(pred_df["영업일자"], pred_df["영업장명_메뉴명"]), pred_df["매출수량"]))
    out = sample.copy()
    for i in out.index:
        date = out.loc[i, "영업일자"]
        for col in out.columns[1:]:
            out.loc[i, col] = pred_dict.get((date, col), 0)
    return out

@torch.no_grad()
def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=cfg.MODEL_NAME)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    fe = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(cfg.TRAIN_FILE, cfg.SEQ_LEN, cfg.HORIZON, cfg.LABEL_LEN,
                              cfg.BATCH_SIZE, fe, num_workers=0)
    dm.prepare_data(); dm.setup("fit")
    item_names = list(dm.target_columns); time_cols = list(dm.time_feature_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, cfg, input_dim=len(item_names)).to(device)

    # ckpt 로딩: Lightning 저장 포맷일 때만 적용(없으면 무시)
    from .core.lightning_module import LitModel
    lit = LitModel(model=model, cfg=cfg, item_names=item_names).to(device)
    try:
        state = torch.load(_find_ckpt(cfg, args.model, args.ckpt), map_location=device)
        lit.load_state_dict(state["state_dict"], strict=False)
    except Exception as e:
        print(f"[warn] ckpt load skipped: {e}")

    results = []
    for path in sorted(glob.glob(os.path.join(cfg.TEST_DIR, "TEST_*.csv"))):
        df = pd.read_csv(path)
        prep = dm.preprocess_inference_data(df)
        if len(prep) < cfg.SEQ_LEN: continue

        x_all = prep[item_names].astype(np.float32)
        x_enc = x_all.tail(cfg.SEQ_LEN).to_numpy().astype(np.float32)
        x_mark_enc = prep[time_cols].tail(cfg.SEQ_LEN).to_numpy().astype(np.float32)

        hist_idx = prep.index[-cfg.LABEL_LEN:]
        fut_idx = pd.date_range(prep.index[-1] + pd.Timedelta(days=1), periods=cfg.HORIZON, freq="D")
        hist_marks = prep.loc[hist_idx, time_cols].astype(np.float32)
        fut_marks = add_date_features(pd.DataFrame({"영업일자": fut_idx}), "영업일자")
        fut_marks.set_index("영업일자", inplace=True)
        fut_marks = fut_marks[time_cols].astype(np.float32)
        y_mark_dec = pd.concat([hist_marks, fut_marks], axis=0).to_numpy(dtype=np.float32)

        batch = {
            "x_enc": torch.from_numpy(x_enc).unsqueeze(0).to(device).float(),
            "x_mark_enc": torch.from_numpy(x_mark_enc).unsqueeze(0).to(device).float(),
            "y_mark_dec": torch.from_numpy(y_mark_dec).unsqueeze(0).to(device).float(),
        }
        yhat = model(batch).squeeze(0).cpu().numpy()  # (H,N)
        yhat = np.clip(np.rint(yhat), 0, None).astype(int)

        test_prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        dates = [f"{test_prefix}+{i+1}일" for i in range(cfg.HORIZON)]
        for t, d in enumerate(dates):
            for j, col in enumerate(item_names):
                results.append({"영업일자": d, "영업장명_메뉴명": col, "매출수량": int(yhat[t, j])})

    pred_long = pd.DataFrame(results)
    sub = _to_submission(pred_long, cfg.SAMPLE_SUBMISSION)
    tag = _param_tag(cfg)
    out = os.path.join(cfg.RESULTS_DIR, f"submission_{args.model}_{tag}.csv")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True); sub.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[ok] saved → {out}")

if __name__ == "__main__":
    main()
