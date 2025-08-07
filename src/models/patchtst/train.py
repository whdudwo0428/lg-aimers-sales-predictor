"""Training script for the PatchTST wrapper.

This script mirrors the structure of ``src/models/fedformer/train.py``
but saves checkpoints under ``checkpoint/patchtst``.  It uses a
placeholder model defined in ``model.py``; replace it with the
official PatchTST implementation as needed.
"""

from __future__ import annotations

import os
from typing import List

import torch

from ...core import data_loader, feature_engineer, data_module, utils
from .model import build_model, ModelConfig


def prepare_features(df) -> List[str]:
    df_fe = feature_engineer.add_date_features(df)
    df_fe = feature_engineer.add_lag_features(df_fe, periods=[1, 7, 14, 21, 28])
    df_fe = feature_engineer.add_moving_average_features(df_fe, windows=[7])
    feature_cols = [c for c in df_fe.columns if c not in {"date", "store_item", "sales"}]
    return df_fe, feature_cols


def main() -> None:
    cfg = ModelConfig()
    utils.seed_everything(cfg.seed)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    train_path = os.path.join(project_root, "dataset", "train")
    df = data_loader.load_train(train_path)
    df_fe, feature_cols = prepare_features(df)
    dm_cfg = data_module.DataModuleConfig(
        input_length=cfg.input_length,
        forecast_horizon=cfg.forecast_horizon,
        batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
    )
    dm = data_module.TimeSeriesDataModule(df=df_fe, feature_columns=feature_cols, target_column="sales", config=dm_cfg)
    dm.setup()
    model = build_model(input_dim=len(feature_cols), config=cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for epoch in range(cfg.max_epochs):
        total_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in dm.train_dataloader():
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{cfg.max_epochs} | Training loss: {avg_loss:.4f}")
    # Save checkpoint
    ckpt_dir = os.path.join(project_root, "checkpoint", "patchtst")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()