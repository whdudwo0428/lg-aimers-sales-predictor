# src/models/model_autoformer.py
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import torch, torch.nn as nn

def _import_original_autoformer():
    root = Path(__file__).resolve().parents[2] / "models" / "Autoformer"
    src_file = root / "models" / "Autoformer.py"
    if not src_file.exists():
        raise ImportError(f"[Autoformer] File not found: {src_file}")
    sys.path.insert(0, str(root))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("autoformer_original", src_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"[Autoformer] Failed to load spec: {src_file}")
    module = ilu.module_from_spec(spec); spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "Model"):
        raise ImportError("[Autoformer] 'Model' not found in Autoformer.py")
    return module.Model

def _build_cfg(cfg: Any, input_dim: int):
    F = cfg.FEDformer  # 기본 하이퍼를 공유(필요 시 별도 섹션으로 분리 가능)
    return SimpleNamespace(
        seq_len=cfg.SEQ_LEN, label_len=cfg.LABEL_LEN, pred_len=cfg.HORIZON,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=F.D_MODEL, n_heads=F.N_HEADS, d_ff=F.D_FF,
        e_layers=F.E_LAYERS, d_layers=F.D_LAYERS, dropout=F.DROPOUT,
        embed=F.EMBED, freq=F.FREQ, activation=F.ACTIVATION,
        output_attention=getattr(F, "OUTPUT_ATTENTION", False),
        moving_avg=getattr(F, "MOVING_AVG", 25), factor=getattr(F, "FACTOR", 1),
    )

class AutoformerModel(nn.Module):
    def __init__(self, original: nn.Module, cfg: Any):
        super().__init__()
        self._orig = original
        self._cfg = cfg
        self._label_len = cfg.LABEL_LEN
        self._horizon = cfg.HORIZON

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> "AutoformerModel":
        Original = _import_original_autoformer()
        ocfg = _build_cfg(cfg, input_dim)
        return cls(Original(ocfg), cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_enc      = batch["x_enc"]       # (B,L,N)
        x_mark_enc = batch["x_mark_enc"]  # (B,L,T_full)
        y_mark_dec = batch["y_mark_dec"]  # (B,Lc+H,T_full)
        B, L, N = x_enc.shape; Lc = self._label_len; H = self._horizon

        # decoder signal: [label_len; zeros]
        x_dec = torch.zeros((B, Lc + H, N), device=x_enc.device, dtype=x_enc.dtype)
        x_dec[:, :Lc, :] = x_enc[:, -Lc:, :]

        # Autoformer도 timeF 임베딩 in_features가 작을 수 있으니 슬라이스
        try:
            t_exp = self._orig.enc_embedding.temporal_embedding.embed.in_features
            if x_mark_enc.size(-1) != t_exp:
                x_mark_enc = x_mark_enc[..., :t_exp]
                y_mark_dec = y_mark_dec[..., :t_exp]
        except Exception:
            pass

        out = self._orig(x_enc, x_mark_enc, x_dec, y_mark_dec)
        if isinstance(out, tuple): out = out[0]
        return out[:, -H:, :]
