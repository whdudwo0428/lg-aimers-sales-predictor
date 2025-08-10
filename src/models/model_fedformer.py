# src/models/model_fedformer.py
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn

from .base_model import BaseModel


def _import_original_fedformer() -> Any:
    """
    Robustly load original FEDformer.Model from the vendored repository
    at <project_root>/models/FEDformer/models/FEDformer.py using a file loader.
    This avoids 'models' package name collisions and __init__.py issues.
    """
    project_root = Path(__file__).resolve().parents[2]  # .../<project_root>
    fed_root = project_root / "models" / "FEDformer"    # .../models/FEDformer
    fed_py = fed_root / "models" / "FEDformer.py"       # .../models/FEDformer/models/FEDformer.py

    if not fed_py.exists():
        raise ImportError(f"[FEDformer] File not found: {fed_py}")

    # Ensure the original repo's root is importable for its internal imports:
    # e.g., 'from layers.Embed import ...', 'from utils.tools import ...'
    sys.path.insert(0, str(fed_root))

    import importlib.util as _ilu
    module_name = "fedformer_original"  # unique name to avoid collisions
    spec = _ilu.spec_from_file_location(module_name, fed_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"[FEDformer] Failed to create import spec for {fed_py}")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    if not hasattr(module, "Model"):
        raise ImportError(f"[FEDformer] 'Model' not found in {fed_py}")
    return module.Model


def _build_original_cfg(cfg: Any, input_dim: int) -> SimpleNamespace:
    """Map our Config to the attributes expected by the original constructor."""
    F = cfg.FEDformer
    return SimpleNamespace(
        # sequence lengths
        seq_len=cfg.SEQ_LEN, label_len=cfg.LABEL_LEN, pred_len=cfg.HORIZON,
        # I/O dims
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        # model size
        d_model=F.D_MODEL, n_heads=F.N_HEADS, d_ff=F.D_FF,
        e_layers=F.E_LAYERS, d_layers=F.D_LAYERS, dropout=F.DROPOUT,
        # time features / embedding
        embed=F.EMBED, freq=F.FREQ,
        # variants / attention
        version=getattr(F, "VERSION", "Fourier"),
        mode_select=getattr(F, "MODE_SELECT", "random"),
        modes=getattr(F, "MODES", 32),
        moving_avg=getattr(F, "MOVING_AVG", 25),
        factor=getattr(F, "FACTOR", 1),
        # wavelet extras (for completeness)
        L=getattr(F, "L", 1),
        base=getattr(F, "BASE", "legendre"),
        cross_activation=getattr(F, "CROSS_ACTIVATION", "tanh"),
        # misc
        activation=F.ACTIVATION,
        output_attention=getattr(F, "OUTPUT_ATTENTION", False),
    )


class FedformerModel(BaseModel):
    """Adapts batch-dict interface → original FEDformer API."""

    def __init__(self, original: nn.Module, cfg: Any) -> None:
        super().__init__()
        self._orig = original
        self._cfg = cfg
        self._label_len = cfg.LABEL_LEN
        self._horizon = cfg.HORIZON

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> "FedformerModel":
        Original = _import_original_fedformer()
        orig_cfg = _build_original_cfg(cfg, input_dim)
        original_model = Original(orig_cfg)
        return cls(original_model, cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch = {
          "x_enc":      (B, seq_len, N),
          "x_mark_enc": (B, seq_len, T),
          "y_mark_dec": (B, label_len + horizon, T)
        } → returns (B, horizon, N)
        """
        x_enc = batch["x_enc"]
        x_mark_enc = batch["x_mark_enc"]
        y_mark_dec = batch["y_mark_dec"]

        # 시간 피처 차원 자동 정합
        x_mark_enc, y_mark_dec = self._slice_time_marks(x_mark_enc, y_mark_dec)

        B, L, N = x_enc.shape
        H, Lc = self._horizon, self._label_len

        # Decoder signal: [last label_len values; zeros for horizon]
        x_dec = torch.zeros((B, Lc + H, N), device=x_enc.device, dtype=x_enc.dtype)
        x_dec[:, :Lc, :] = x_enc[:, -Lc:, :]

        out = self._orig(x_enc, x_mark_enc, x_dec, y_mark_dec)
        if isinstance(out, tuple):  # (y_hat, attn)
            out = out[0]
        if out.shape[1] != H:
            out = out[:, -H:, :]
        return out

    def _slice_time_marks(self, x_mark_enc: torch.Tensor, y_mark_dec: torch.Tensor):
        """
        원본 FEDformer enc_embedding의 temporal_embedding이 기대하는
        입력 차원(in_features)을 런타임에 탐지하고, 그에 맞춰 x_mark/y_mark를 슬라이스.
        """
        t_exp = None
        try:
            emb = getattr(self._orig, "enc_embedding", None)
            if emb is not None:
                # 흔한 네이밍을 순회하며 Linear를 찾는다.
                for cand in ("temporal_embedding", "temporal_embed", "time_embedding"):
                    te = getattr(emb, cand, None)
                    if te is None:
                        continue
                    for attr in ("embed", "linear", "proj", "projection", "projection_layer"):
                        lin = getattr(te, attr, None)
                        if isinstance(lin, nn.Linear):
                            t_exp = lin.in_features
                            break
                    if t_exp is not None:
                        break
        except Exception:
            t_exp = None

        # 탐지가 안 되면 FREQ 기준의 안전한 폴백(일 단위면 3)
        if t_exp is None:
            freq = str(getattr(getattr(self._cfg, "FEDformer", None), "FREQ", "d")).lower()
            if freq.startswith("d"):  # day
                t_exp = 3
            elif freq.startswith("h"):  # hour
                t_exp = 4
            else:
                t_exp = min(x_mark_enc.size(-1), 4)

        if x_mark_enc.size(-1) != t_exp:
            x_mark_enc = x_mark_enc[..., :t_exp]
            y_mark_dec = y_mark_dec[..., :t_exp]
        return x_mark_enc, y_mark_dec