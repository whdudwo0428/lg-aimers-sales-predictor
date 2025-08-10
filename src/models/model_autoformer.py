from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import torch
import torch.nn as nn

def _import_original_autoformer():
    root = Path(__file__).resolve().parents[2] / "models" / "Autoformer"
    src = root / "models" / "Autoformer.py"
    if not src.exists():
        raise ImportError(f"[Autoformer] Not found: {src}")
    sys.path.insert(0, str(root))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("autoformer_original", src)
    if not spec or not spec.loader:
        raise ImportError(f"[Autoformer] load spec failed: {src}")
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "Model"):
        raise ImportError("[Autoformer] 'Model' not found")
    return mod.Model

def _build_cfg(cfg: Any, input_dim: int):
    F = getattr(cfg, "FEDformer")  # 기본 하이퍼 공유
    A = getattr(cfg, "Autoformer", None) or F
    return SimpleNamespace(
        seq_len=cfg.SEQ_LEN, label_len=cfg.LABEL_LEN, pred_len=cfg.HORIZON,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=getattr(A, "D_MODEL", F.D_MODEL),
        n_heads=getattr(A, "N_HEADS", F.N_HEADS),
        e_layers=getattr(A, "E_LAYERS", F.E_LAYERS),
        d_layers=getattr(A, "D_LAYERS", F.D_LAYERS),
        d_ff=getattr(A, "D_FF", F.D_FF),
        dropout=getattr(A, "DROPOUT", F.DROPOUT),
        embed=getattr(A, "EMBED", F.EMBED),
        freq=getattr(A, "FREQ", F.FREQ),
        activation=getattr(A, "ACTIVATION", F.ACTIVATION),
        output_attention=getattr(A, "OUTPUT_ATTENTION", False),
        moving_avg=getattr(A, "MOVING_AVG", 25),
        factor=getattr(A, "FACTOR", 1),
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

    def _slice_time_marks(self, x_mark_enc: torch.Tensor, y_mark_dec: torch.Tensor):
        import torch.nn as nn
        t_exp = None
        try:
            emb = getattr(self._orig, "enc_embedding", None)
            te = getattr(emb, "temporal_embedding", None)
            lin = getattr(te, "embed", None)
            if isinstance(lin, nn.Linear):
                t_exp = lin.in_features
        except Exception:
            t_exp = None
        if t_exp is None:  # 안전 폴백
            t_exp = 3 if str(getattr(self._cfg.FEDformer, "FREQ", "d")).lower().startswith("d") else min(x_mark_enc.size(-1), 4)
        if x_mark_enc.size(-1) != t_exp:
            x_mark_enc = x_mark_enc[..., :t_exp]
            y_mark_dec = y_mark_dec[..., :t_exp]
        return x_mark_enc, y_mark_dec

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_enc = batch["x_enc"]
        x_mark_enc = batch.get("x_mark_enc", batch.get("x_mark"))
        y_mark_dec = batch.get("y_mark_dec", batch.get("y_mark"))
        if x_mark_enc.dtype != x_enc.dtype: x_mark_enc = x_mark_enc.to(x_enc.dtype)
        if y_mark_dec.dtype != x_enc.dtype: y_mark_dec = y_mark_dec.to(x_enc.dtype)

        x_mark_enc, y_mark_dec = self._slice_time_marks(x_mark_enc, y_mark_dec)

        B, L, N = x_enc.shape; H, Lc = self._horizon, self._label_len
        x_dec = torch.zeros((B, Lc + H, N), device=x_enc.device, dtype=x_enc.dtype)
        x_dec[:, :Lc, :] = x_enc[:, -Lc:, :]

        out = self._orig(x_enc, x_mark_enc, x_dec, y_mark_dec)
        if isinstance(out, tuple): out = out[0]
        return out[:, -H:, :]
