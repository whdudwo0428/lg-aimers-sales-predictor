# src/models/model_patchtst.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict
import torch, torch.nn as nn

def _import_patchtst_impl():
    """
    두 경로 중 하나를 사용:
    1) 로컬 리포: <project_root>/models/PatchTST/PatchTST.py
    2) 설치 패키지: import patchtst
    """
    # 1) 로컬 파일
    root = Path(__file__).resolve().parents[2] / "models" / "PatchTST"
    src_file = root / "PatchTST.py"
    if src_file.exists():
        sys.path.insert(0, str(root))
        import importlib.util as ilu
        spec = ilu.spec_from_file_location("patchtst_original", src_file)
        if spec and spec.loader:
            module = ilu.module_from_spec(spec); spec.loader.exec_module(module)  # type: ignore
            if hasattr(module, "PatchTST"):  # 원본 구현 클래스명에 맞춰 조정
                return module.PatchTST
    # 2) 설치 패키지
    try:
        from patchtst import PatchTST  # type: ignore
        return PatchTST
    except Exception as e:
        raise ImportError("[PatchTST] 구현을 찾을 수 없습니다. models/PatchTST/ 또는 패키지 설치를 확인하세요.") from e

class PatchTSTModel(nn.Module):
    def __init__(self, original: nn.Module, cfg: Any):
        super().__init__()
        self._orig = original
        self._horizon = cfg.HORIZON

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> "PatchTSTModel":
        Impl = _import_patchtst_impl()
        # 원본 구현 시그니처에 맞춰 전달(필요 시 조정)
        model = Impl(c_in=input_dim, pred_len=cfg.HORIZON, seq_len=cfg.SEQ_LEN)
        return cls(model, cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x_enc"]                   # (B,L,N)
        B, L, N = x.shape
        # PatchTST는 (B,N,L) 관례
        x = x.permute(0, 2, 1).contiguous()
        # 간단 표준화(옵션)
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        y = self._orig(x)                    # (B,N,H) 가정 (구현에 맞게 조정)
        if y.dim() == 3 and y.shape[1] == N:
            y = y.permute(0, 2, 1).contiguous()  # (B,H,N)
        return y
