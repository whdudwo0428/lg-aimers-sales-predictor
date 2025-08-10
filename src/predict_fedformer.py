"""
Entry point for generating predictions with the FEDformer wrapper.

This script loads the most recent checkpoint from
``results/checkpoints``, reconstructs the model and associated data
module and produces forecasts for each test batch (``TEST_00.csv`` …
``TEST_09.csv``).  Individual predictions for each test file are
stored under ``results/preds/fedformer`` and the final merged
submission is written to ``results/submission.csv``.

Usage
-----
Run the following command from the project root::

    python -m src.models.predict_fedformer

Prerequisites
-------------
* A trained model checkpoint must exist in ``results/checkpoints``.  If
  multiple checkpoints are present the most recently modified file is
  used.
* The test files must be located in ``dataset/test`` and follow the
  naming convention ``TEST_XX.csv``.
* The sample submission must be available at
  ``dataset/sample_submission.csv``.
"""

from __future__ import annotations

import os
import glob
from typing import List, Dict

import pandas as pd
import numpy as np
from ..config import Config
from ..core.data_module import TimeSeriesDataModule
from ..core.feature_engineer import FeatureEngineer
from ..core.utils import seed_everything
from .model_fedformer import FedformerModel

# We avoid importing torch and pytorch_lightning at module load time
# because these libraries may not be installed in the execution
# environment.  Instead, we import them within the predict() function.


def load_latest_checkpoint(checkpoint_dir: str) -> str:
    """Return the path to the most recently modified checkpoint file."""
    ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]


def predict() -> None:
    """Run inference using the most recently saved checkpoint.

    This function loads the latest checkpoint, reconstructs the model
    and data module, generates predictions for each test batch and
    writes both per‑file and merged submission outputs.  Imports of
    torch and PyTorch Lightning occur inside this function to
    gracefully handle environments where they are unavailable.  If
    these libraries are missing the function will emit a message and
    return without performing inference.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[predict_fedformer] PyTorch is not installed. Skipping prediction.")
        return

    from ..core.lightning_module import LitModel  # import lazily
    # Attempt to import optional PyTorch Lightning, but we don't strictly need it here
    try:
        import pytorch_lightning as pl  # noqa: F401
    except ImportError:
        # Continue without lightning; we still need the checkpoint loader defined on LitModel
        pass

    cfg = Config()
    seed_everything(cfg.SEED)

    # Locate checkpoint
    best_ckpt = load_latest_checkpoint(cfg.CHECKPOINT_DIR)
    print(f"Loading checkpoint: {best_ckpt}")

    # Prepare data module (training data only to define pivot and features)
    fe = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(
        file_path=cfg.TRAIN_FILE,
        sequence_length=cfg.SEQ_LEN,
        forecast_horizon=cfg.HORIZON,
        label_len=cfg.LABEL_LEN,
        batch_size=cfg.BATCH_SIZE,
        feature_engineer=fe,
        num_workers=cfg.NUM_WORKERS,
    )
    dm.prepare_data()

    input_dim = dm.input_dim
    item_names = list(dm.target_columns)

    # Instantiate model and LightningModule, then load weights
    model = FedformerModel.from_config(cfg, input_dim=input_dim)
    lit_model = LitModel(model=model, cfg=cfg, item_names=item_names)
    # Load from checkpoint; pass through model and cfg to satisfy constructor
    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path=best_ckpt,
        model=model,
        cfg=cfg,
        item_names=item_names,
    )
    lit_model.eval()
    lit_model.freeze()

    # Device placement: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model.to(device)

    preds_dir = os.path.join(cfg.PREDS_DIR, "fedformer")
    os.makedirs(preds_dir, exist_ok=True)

    all_pred_rows: List[Dict[str, any]] = []

    test_files = sorted(glob.glob(os.path.join(cfg.TEST_DIR, "TEST_*.csv")))
    for path in test_files:
        df_test = pd.read_csv(path)
        prefix = os.path.splitext(os.path.basename(path))[0]
        prepared = dm.preprocess_inference_data(df_test)
        seqs = dm._create_sequences(prepared)
        if seqs[0].size == 0:
            print(f"Warning: no sequences created for {path}; skipping this file.")
            continue
        x_enc_np, y_seq_np, x_mark_np, y_mark_np = seqs
        x_enc = torch.from_numpy(x_enc_np[-1:]).to(device)
        x_mark = torch.from_numpy(x_mark_np[-1:]).to(device)
        y_mark = torch.from_numpy(y_mark_np[-1:]).to(device)
        batch_dict = {
            "x_enc": x_enc,
            "x_mark_enc": x_mark,
            "y_mark_dec": y_mark,
        }
        with torch.no_grad():
            pred = lit_model.model(batch_dict)
        pred_np = pred.squeeze(0).cpu().numpy()
        pred_np = np.clip(pred_np, 0, None)
        pred_np = np.rint(pred_np).astype(int)
        horizon = pred_np.shape[0]
        for i in range(horizon):
            date_str = f"{prefix}+{i+1}일"
            for item_idx, item_name in enumerate(item_names):
                value = int(pred_np[i, item_idx])
                all_pred_rows.append({
                    "영업일자": date_str,
                    "영업장명_메뉴명": item_name,
                    "매출수량": value,
                })
        file_pred_df = pd.DataFrame([
            {"영업일자": f"{prefix}+{i+1}일", **{item_names[j]: pred_np[i, j] for j in range(len(item_names))}}
            for i in range(horizon)
        ])
        file_pred_path = os.path.join(preds_dir, f"{prefix}.csv")
        file_pred_df.to_csv(file_pred_path, index=False, encoding="utf-8-sig")
        print(f"Saved predictions for {prefix} to {file_pred_path}")

    # Assemble full prediction DataFrame
    sample = pd.read_csv(cfg.SAMPLE_SUBMISSION)
    pred_dict = {(row["영업일자"], row["영업장명_메뉴명"]): row["매출수량"] for row in all_pred_rows}
    final_df = sample.copy()
    for idx in final_df.index:
        date = final_df.loc[idx, "영업일자"]
        for col in final_df.columns[1:]:
            final_df.loc[idx, col] = pred_dict.get((date, col), 0)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    final_df.to_csv(cfg.SUBMISSION_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved merged submission to {cfg.SUBMISSION_FILE}")


if __name__ == "__main__":
    predict()