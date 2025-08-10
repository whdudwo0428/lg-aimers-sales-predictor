# src/models/factory.py
from __future__ import annotations
from typing import Any
from .model_fedformer import FedformerModel

# 필요한 시점에 원본 리포를 배치/설치하면 아래 임포트가 정상화됩니다.
try:
    from .model_autoformer import AutoformerModel
except Exception:
    AutoformerModel = None  # 원본 리포 추가 전까지는 None으로 둠

try:
    from .model_patchtst import PatchTSTModel
except Exception:
    PatchTSTModel = None

try:
    from .model_mamba import MambaTSModel
except Exception:
    MambaTSModel = None

REGISTRY = {
    "fedformer":  FedformerModel,
    "autoformer": AutoformerModel,
    "patchtst":   PatchTSTModel,
    "mamba":      MambaTSModel,
}

def build_model(name: str, cfg: Any, input_dim: int):
    name = (name or "fedformer").lower()
    if name not in REGISTRY or REGISTRY[name] is None:
        raise ImportError(
            f"[factory] Model '{name}' is not available. "
            f"원본 리포 배치 또는 해당 래퍼 파일 확인이 필요합니다."
        )
    return REGISTRY[name].from_config(cfg, input_dim)
