# src/train_fedformer.py
"""
Entry point for training the FEDformer model (moved to ``src/``).

Run from project root:

    python -m src.train_fedformer
"""
from __future__ import annotations

import os
from typing import Any

from .config import Config
from .core.data_module import TimeSeriesDataModule
from .core.feature_engineer import FeatureEngineer
from .core.utils import seed_everything
from .models.model_fedformer import FedformerModel


def train() -> None:
    # Import torch/PL lazily so this file can be imported without them.
    try:
        import torch  # noqa: F401
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    except Exception as e:
        print(f"[train_fedformer] Missing dependency: {e}")
        return

    cfg = Config()
    seed_everything(cfg.SEED)

    # -----------------
    # Data
    # -----------------
    feature_engineer = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(
        file_path=cfg.TRAIN_FILE,
        sequence_length=cfg.SEQ_LEN,
        forecast_horizon=cfg.HORIZON,
        label_len=cfg.LABEL_LEN,
        batch_size=cfg.BATCH_SIZE,
        feature_engineer=feature_engineer,
        num_workers=cfg.NUM_WORKERS,
    )
    dm.prepare_data()
    dm.setup("fit")

    # -----------------
    # Model
    # -----------------
    input_dim = dm.input_dim  # number of series (columns)
    model = FedformerModel.from_config(cfg, input_dim=input_dim)

    # LightningModule (loss, logging, optim)
    from .core.lightning_module import LitModel
    item_names = list(dm.target_columns)
    lit_model = LitModel(model=model, cfg=cfg, item_names=item_names)

    # -----------------
    # Trainer
    # -----------------
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    tag = f"d{cfg.FEDformer.D_MODEL}_L{cfg.FEDformer.E_LAYERS}_seq{cfg.SEQ_LEN}_h{cfg.HORIZON}_bs{cfg.BATCH_SIZE}_lr{cfg.LR}"
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=cfg.CHECKPOINT_DIR,
        filename=f"fedformer_{tag}",
        save_top_k=1,
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.PATIENCE, verbose=True)

    trainer = pl.Trainer(
        max_epochs = cfg.EPOCHS,
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION,
        gradient_clip_val=getattr(cfg, "GRAD_CLIP", 0.0),
        callbacks=[ckpt_cb, es_cb],
        default_root_dir=cfg.RESULTS_DIR,
        log_every_n_steps=cfg.LOG_EVERY_N_STEPS,
    )

    print("--- Starting training ---")
    trainer.fit(lit_model, dm)
    print("--- Training completed ---")

    if ckpt_cb.best_model_path:
        print(f"Best model saved to: {ckpt_cb.best_model_path}")
    else:
        print("No checkpoint was saved.")


if __name__ == "__main__":
    train()
