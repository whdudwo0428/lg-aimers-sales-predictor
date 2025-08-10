"""
PyTorch Lightning module wrapper for training forecasting models.

This module adapts an arbitrary model implementing a ``forward`` method
to the PyTorch Lightning API.  It handles computing the Weighted
SMAPE loss, logging metrics and configuring the optimiser and
learning rate scheduler.  The module expects the input batches to be
tuples containing four tensors: ``(x_enc, y_seq, x_mark, y_mark)``.
These correspond to the encoder inputs, decoder targets, encoder
calendar features and decoder calendar features, respectively.
"""

from __future__ import annotations

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Tuple

from .loss import weighted_smape_loss


class LitModel(pl.LightningModule):
    """Lightning wrapper that trains a forecasting model using Weighted SMAPE.

    Parameters
    ----------
    model : torch.nn.Module
        The underlying PyTorch model.  Its ``forward`` method must
        accept a dictionary with keys ``x_enc``, ``x_mark_enc`` and
        optionally ``y_mark_dec``, and return a tensor of shape
        ``(B, horizon, N)`` where ``N`` is the number of series.  Any
        attention weights returned by the model are ignored.
    cfg : Config
        Configuration object containing optimisation hyperparameters.
    item_names : Iterable[str]
        Names of the series in the order they appear in the data.  Used
        to construct the per‑item weight vector.
    device : torch.device or str, optional
        Desired device.  The weights tensor will be moved to this
        device.  If omitted the trainer will place the module on the
        appropriate device.
    """

    def __init__(self, model: torch.nn.Module, cfg: Any, item_names: Any, device: Any = None) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

        # Compute weights: 2 for items containing Damha or Miracia, else 1
        weight_list = []
        for name in item_names:
            if ("Damha" in name) or ("Miracia" in name):
                weight_list.append(2.0)
            else:
                weight_list.append(1.0)
        weight_tensor = torch.tensor(weight_list, dtype=torch.float32)
        if device is not None:
            weight_tensor = weight_tensor.to(device)
        # Register as buffer so that it moves with the model
        self.register_buffer("weights", weight_tensor)

        # Expose horizon and label length for convenience
        self.horizon = cfg.HORIZON
        self.label_len = cfg.LABEL_LEN

        # Log configuration summary once at the start of training
        self.logged_hyperparams = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the underlying model.

        The model is expected to accept a dictionary containing
        ``x_enc``, ``x_mark_enc`` and optionally ``y_mark_dec``.  The
        ``y_mark_dec`` argument may be omitted by models that ignore
        decoder features.
        """
        return self.model(batch)

    # ------------------------------------------------------------------
    # Training/validation/test steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        # Unpack the tuple produced by the DataLoader
        x_enc, y_seq, x_mark, y_mark = batch
        # Construct the decoder context and true values
        # y_seq has shape (B, label_len + horizon, N)
        # The model receives the decoder context (label_len timesteps) via y_mark
        batch_dict = {
            "x_enc": x_enc,
            "x_mark_enc": x_mark,
            "y_mark_dec": y_mark,
        }
        # Forward pass returns predictions of shape (B, horizon, N)
        pred = self(batch_dict)
        # Extract the ground truth horizon portion from y_seq
        true = y_seq[:, -self.horizon :, :]
        # Compute Weighted SMAPE
        loss = weighted_smape_loss(pred, true, self.weights)
        # Logging
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if not self.logged_hyperparams:
            # 문자열은 log 불가 → 숫자만 기록, 디바이스 문자열은 print로만
            dev = self.device  # Lightning이 보장하는 현재 모듈 디바이스
            is_cuda = 1 if getattr(dev, "type", None) == "cuda" else 0
            self.log("is_cuda", is_cuda, prog_bar=False, logger=True)
            # (선택) 텍스트는 콘솔에만
            self.print(f"[device] {dev}")
            self.log("seed", self.cfg.SEED, prog_bar=False, logger=True)
            self.logged_hyperparams = True
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, stage="test")

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        # Use AdamW as the optimiser
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WD
        )
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.cfg.EPOCHS
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


__all__ = ["LitModel"]