from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import torch
import torch.nn as nn

def _import_original_patchtst():
    # PatchTST_supervised/models/PatchTST.py + relative 'layers/..' 임포트 필요
    root = Path(__file__).resolve().parents[2] / "models" / "PatchTST" / "PatchTST_supervised"
    src = root / "models" / "PatchTST.py"
    if not src.exists():
        raise ImportError(f"[PatchTST] Not found: {src}")
    sys.path.insert(0, str(root))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("patchtst_original", src)
    if not spec or not spec.loader:
        raise ImportError(f"[PatchTST] load spec failed: {src}")
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    if hasattr(mod, "Model"): return mod.Model
    if hasattr(mod, "PatchTST"): return mod.PatchTST
    raise ImportError("[PatchTST] Model class not found in PatchTST.py")

def _build_cfg(cfg: Any, input_dim: int):
    P = getattr(cfg, "PatchTST", None)
    F = getattr(cfg, "FEDformer")
    # 기본값: 합리적 안전 파라미터 세트
    return SimpleNamespace(
        enc_in=input_dim, seq_len=cfg.SEQ_LEN, pred_len=cfg.HORIZON,
        e_layers=getattr(P, "E_LAYERS", 3),
        d_model=getattr(P, "D_MODEL", F.D_MODEL),
        n_heads=getattr(P, "N_HEADS", F.N_HEADS),
        d_ff=getattr(P, "D_FF", 4 * getattr(P, "D_MODEL", F.D_MODEL)),
        dropout=getattr(P, "DROPOUT", 0.2),
        revin=getattr(P, "REVIN", True),
        affine=getattr(P, "AFFINE", False),
        individual=getattr(P, "INDIVIDUAL", False),
        fc_dropout=getattr(P, "FC_DROPOUT", 0.0),
        head_dropout=getattr(P, "HEAD_DROPOUT", 0.0),
        patch_len=getattr(P, "PATCH_LEN", 16),
        stride=getattr(P, "STRIDE", 8),
        padding_patch=getattr(P, "PADDING_PATCH", "end"),
        decomposition=getattr(P, "DECOMPOSITION", True),
        kernel_size=getattr(P, "KERNEL_SIZE", 25),
        subtract_last=getattr(P, "SUBTRACT_LAST", False),
    )

class PatchTSTModel(nn.Module):
    def __init__(self, original: nn.Module, cfg: Any):
        super().__init__()
        self._orig = original
        self._horizon = cfg.HORIZON

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> "PatchTSTModel":
        Original = _import_original_patchtst()
        ocfg = _build_cfg(cfg, input_dim)
        return cls(Original(ocfg), cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # PatchTST는 time mark 사용 X → x_enc만 전달
        x = batch["x_enc"]  # (B, L, N)
        y = self._orig(x)   # 일반적으로 (B, H, N)
        if isinstance(y, tuple): y = y[0]
        # 혹시 (B, N, H)로 나오면 (B, H, N)로 전치
        if y.dim() == 3 and y.shape[1] == x.shape[2] and y.shape[2] == self._horizon:
            y = y.permute(0, 2, 1).contiguous()
        return y
