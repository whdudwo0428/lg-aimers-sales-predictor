"""Training script for the FedFormer wrapper.

This module provides a minimal PyTorch training loop that reads the
training data, applies basic feature engineering, constructs a
``TimeSeriesDataModule`` from the processed DataFrame and trains a
simple forecasting model.  The trained model's weights are saved to
``checkpoint/fedformer/best.ckpt`` at the end of training.

Note that this script intentionally avoids PyTorch Lightning to keep
dependencies minimal.  Should you prefer Lightning, you can adapt
this to use ``pl.Trainer`` with the ``TimeSeriesDataModule`` and
appropriate callbacks.
"""

from __future__ import annotations

import os
from typing import List

import torch

from ...core import data_loader, feature_engineer, data_module, utils
from .model import build_model, ModelConfig


def prepare_features(df) -> List[str]:
    """Apply feature engineering to the raw training dataframe.

    Adds date features, a set of fixed lag features and a simple
    moving average.  Returns the list of newly created feature
    column names.
    """
    df_fe = feature_engineer.add_date_features(df)
    # Hard‑coded lag periods and moving average window; customise as needed
    df_fe = feature_engineer.add_lag_features(df_fe, periods=[1, 7, 14, 21, 28])
    df_fe = feature_engineer.add_moving_average_features(df_fe, windows=[7])
    # Identify feature columns: everything except these
    feature_cols = [c for c in df_fe.columns if c not in {"date", "store_item", "sales"}]
    return df_fe, feature_cols


def main() -> None:
    # Load configuration – for a real project you might read from a YAML
    cfg = ModelConfig()
    utils.seed_everything(cfg.seed)
    # Load raw training data
    # Compute project root: src/models/fedformer/train.py -> models/fedformer -> models -> src -> lg-project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    train_path = os.path.join(project_root, "dataset", "train")
    df = data_loader.load_train(train_path)
    # Apply feature engineering
    df_fe, feature_cols = prepare_features(df)
    # Construct DataModule
    dm_cfg = data_module.DataModuleConfig(
        input_length=cfg.input_length,
        forecast_horizon=cfg.forecast_horizon,
        batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
    )
    dm = data_module.TimeSeriesDataModule(
        df=df_fe,
        feature_columns=feature_cols,
        target_column="sales",
        config=dm_cfg,
    )
    dm.setup()
    # Build model
    model = build_model(input_dim=len(feature_cols), config=cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.MSELoss()
    # Training loop
    model.train()
    for epoch in range(cfg.max_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in dm.train_dataloader():
            optimizer.zero_grad()
            preds = model(X_batch)
            # y_batch has shape (batch, horizon, 1) – squeeze last dim
            loss = loss_fn(preds, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{cfg.max_epochs} | Training loss: {avg_loss:.4f}")
    # Save checkpoint
    ckpt_dir = os.path.join(project_root, "checkpoint", "fedformer")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()