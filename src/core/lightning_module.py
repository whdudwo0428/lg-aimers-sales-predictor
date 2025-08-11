from __future__ import annotations

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
from types import SimpleNamespace

from .loss import weighted_smape_loss


class LitModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg: Any, item_names: Any, device: Any = None) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

        # item별 가중치
        w = []
        for name in item_names:
            w.append(2.0 if ("Damha" in name or "Miracia" in name) else 1.0)
        wt = torch.tensor(w, dtype=torch.float32)
        if device is not None:
            wt = wt.to(device)
        self.register_buffer("weights", wt)

        self.horizon = cfg.HORIZON
        self.label_len = cfg.LABEL_LEN
        self.logged_hyperparams = False

    # ------------------------------ 배치 유틸 ------------------------------
    def _build_x_dec(self, y_seq: torch.Tensor) -> torch.Tensor:
        """
        디코더 입력 생성:
        - teacher forcing 없이 일반적으로 label_len 구간(y_seq 앞부분)을 컨텍스트로 주고
          horizon 길이만큼 0을 붙여 디코더 입력을 만든다.
        """
        dec_zeros = torch.zeros_like(y_seq[:, -self.horizon:, :])
        return torch.cat([y_seq[:, :self.label_len, :], dec_zeros], dim=1)

    def _format_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        다양한 형태의 배치를 표준 딕셔너리로 통일:
        반환 키: x_enc, x_mark_enc, x_dec, y_mark_dec, y_true
        """
        # 1) 이미 dict인 경우
        if isinstance(batch, dict):
            b = dict(batch)  # copy
            # y_true가 없지만 y_seq가 있으면 만들어 준다
            if "y_true" not in b and "y_seq" in b:
                b["y_true"] = b["y_seq"][:, -self.horizon:, :]
            # x_dec이 없고 y_seq가 있으면 생성
            if "x_dec" not in b and "y_seq" in b:
                b["x_dec"] = self._build_x_dec(b["y_seq"])
            return b

        # 2) (dict,) 래핑
        if isinstance(batch, (tuple, list)) and len(batch) == 1 and isinstance(batch[0], dict):
            return self._format_batch(batch[0])

        # 3) (x_enc, y_seq, x_mark, y_mark) 튜플
        if isinstance(batch, (tuple, list)) and len(batch) == 4 and all(torch.is_tensor(t) for t in batch):
            x_enc, y_seq, x_mark, y_mark = batch
            return {
                "x_enc": x_enc,
                "x_mark_enc": x_mark,
                "x_dec": self._build_x_dec(y_seq),
                "y_mark_dec": y_mark,
                "y_true": y_seq[:, -self.horizon:, :],
            }

        raise TypeError(f"Unsupported batch type for formatting: {type(batch)}")

    def format_batch(self, batch) -> Dict[str, torch.Tensor]:
        return self._format_batch(batch)

    # ------------------------------ forward ------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 모델은 dict를 그대로 받도록 통일 (FEDformer/Autoformer/PatchTST 래퍼가 키를 읽음)
        return self.model(batch)

    # ------------------------- train/val/test step ------------------------
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        b = self._format_batch(batch)
        pred = self(b)
        true = b.get("y_true", None)
        if true is None:
            raise RuntimeError("y_true is missing in batch after formatting.")
        loss = weighted_smape_loss(pred, true, self.weights)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if not self.logged_hyperparams:
            dev = self.device
            self.log("is_cuda", 1 if getattr(dev, "type", None) == "cuda" else 0, prog_bar=False, logger=True)
            self.print(f"[device] {dev}")
            self.log("seed", self.cfg.SEED, prog_bar=False, logger=True)
            self.logged_hyperparams = True
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, stage="test")

    # ------------------------- optimizer/scheduler ------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        lr = float(getattr(self.cfg, "LR", getattr(self.cfg, "LEARNING_RATE", 1e-3)))
        wd = getattr(self.cfg, "WD", getattr(self.cfg, "WEIGHT_DECAY", 0.0))
        wd = float(wd)

        optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        epochs = int(getattr(self.cfg, "EPOCHS", 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"},
        }
