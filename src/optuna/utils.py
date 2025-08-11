from __future__ import annotations
import gc
import re
import torch
from typing import Dict, Any
from types import SimpleNamespace



def _sanitize_key(k: str) -> str:
    """
    'fedformer.n_heads@256#r2' -> ('FEDformer', 'N_HEADS')
    'fedformer.d_model'       -> ('FEDformer', 'D_MODEL')
    'BATCH_SIZE'              -> (None, 'BATCH_SIZE')
    """
    if "." not in k:
        return None, k.upper()

    model, name = k.split(".", 1)
    # '@', '#'(tag) 등 suffix 제거
    name = re.split(r"[@#]", name)[0]
    return model.upper(), name.upper()

def _ensure_ns(obj, name: str):
    if not hasattr(obj, name):
        setattr(obj, name, SimpleNamespace())
    return getattr(obj, name)

def apply_overrides(base_cfg, overrides: dict):
    """
    overrides 딕셔너리를 Config에 반영.
    - 대문자/소문자 구분 없이 처리
    - 'model.field' 형태는 하위 네임스페이스로 반영 (예: FEDformer.D_MODEL)
    - 알 수 없는 키도 허용(경고 출력)하고 동적으로 추가
    """
    for raw_key, value in overrides.items():
        key = raw_key.strip()

        if "." in key:
            # ex) "fedformer.d_model"
            top, sub = key.split(".", 1)
            top_u = top.upper()
            # 하위 네임스페이스 보장
            ns = _ensure_ns(base_cfg, top_u)
            # 하위 키를 대문자로 통일
            setattr(ns, sub.upper(), value)
        else:
            top_u = key.upper()
            if not hasattr(base_cfg, top_u):
                print(f"[apply_overrides] WARN: '{top_u}' not in Config. Creating dynamically.")
            setattr(base_cfg, top_u, value)

    return base_cfg


def rough_vram_score(batch_size: int, seq_len: int, d_model: int, n_heads: int) -> int:
    """VRAM 대략 추정: 배치 x 시퀀스 x 차원 x 헤드 (프루닝용 지표)."""
    return int(batch_size) * int(seq_len) * int(d_model) * int(n_heads)


def cleanup_cuda():
    """CUDA/CPU 공통 메모리 정리."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
