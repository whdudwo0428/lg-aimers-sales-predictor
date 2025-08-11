# src/train_any.py
from __future__ import annotations
import os
import json, argparse
from .config import Config
from .core.feature_engineer import FeatureEngineer
from .core.data_module import TimeSeriesDataModule
from .core.utils import seed_everything
from .models.factory import build_model

def train() -> None:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    cfg = Config()
    # (옵션) Optuna 결과 덮어쓰기
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--override", type=str, default=None,
                    help="dot-override JSON 경로 (e.g., results/optuna/fedformer/best_config_overrides.json)")
    args, _ = ap.parse_known_args()
    if args.override:
        from .optuna.utils import apply_overrides
        with open(args.override, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        cfg = apply_overrides(cfg, overrides)

    seed_everything(cfg.SEED)

    fe = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(
        file_path=cfg.TRAIN_FILE, sequence_length=cfg.SEQ_LEN,
        forecast_horizon=cfg.HORIZON, label_len=cfg.LABEL_LEN,
        batch_size=cfg.BATCH_SIZE, feature_engineer=fe, num_workers=cfg.NUM_WORKERS,
    )
    dm.prepare_data(); dm.setup("fit")

    input_dim = dm.input_dim
    model = build_model(cfg.MODEL_NAME, cfg, input_dim)

    from .core.lightning_module import LitModel
    lit = LitModel(model=model, cfg=cfg, item_names=list(dm.target_columns))

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min",
                           dirpath=cfg.CHECKPOINT_DIR,
                           filename=f"{cfg.MODEL_NAME}_d{cfg.FEDformer.D_MODEL}_L{cfg.FEDformer.E_LAYERS}_seq{cfg.SEQ_LEN}_h{cfg.HORIZON}_bs{cfg.BATCH_SIZE}_lr{cfg.LR}",
                           save_top_k=1)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.PATIENCE, verbose=True)

    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION,
        gradient_clip_val=getattr(cfg, "GRAD_CLIP", 0.0),
        callbacks=[ckpt, es],
        default_root_dir=cfg.RESULTS_DIR,
        log_every_n_steps=cfg.LOG_EVERY_N_STEPS,
    )
    print(f"--- Starting training ({cfg.MODEL_NAME}) ---")
    trainer.fit(lit, dm)
    if ckpt.best_model_path:
        print(f"Best model saved to: {ckpt.best_model_path}")

if __name__ == "__main__":
    train()
