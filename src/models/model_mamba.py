# src/models/model_mamba.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict
import torch, torch.nn as nn

def _import_mamba_impl():
    """
    로컬 리포: <project_root>/models/MambaTS/MambaTS.py 또는 설치 패키지.
    """
    root = Path(__file__).resolve().parents[2] / "models" / "MambaTS"
    src_file = root / "MambaTS.py"
    if src_file.exists():
        sys.path.insert(0, str(root))
        import importlib.util as ilu
        spec = ilu.spec_from_file_location("mamba_ts_original", src_file)
        if spec and spec.loader:
            module = ilu.module_from_spec(spec); spec.loader.exec_module(module)  # type: ignore
            if hasattr(module, "MambaTS"):
                return module.MambaTS
    try:
        from mambats import MambaTS  # 예시 패키지명, 실제에 맞게 수정
        return MambaTS
    except Exception as e:
        raise ImportError("[Mamba] 구현을 찾을 수 없습니다. models/MambaTS/ 또는 패키지 설치를 확인하세요.") from e

class MambaTSModel(nn.Module):
    def __init__(self, original: nn.Module, cfg: Any):
        super().__init__()
        self._orig = original

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> "MambaTSModel":
        Impl = _import_mamba_impl()
        model = Impl(c_in=input_dim, pred_len=cfg.HORIZON, seq_len=cfg.SEQ_LEN)
        return cls(model, cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x_enc"]  # (B,L,N)
        return self._orig(x)  # (B,H,N) 형태로 구현되어 있어야 함
