# src/models/model_fedformer.py
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn

from .base_model import BaseModel

def _safe_modes_cap(cfg) -> int:
    """enc/dec에서 실제 FFT 축 길이를 고려해 modes 상한을 계산"""
    enc_nfreq = max(1, int(cfg.SEQ_LEN) // 2)
    dec_len = int(cfg.LABEL_LEN) + int(cfg.HORIZON)
    dec_nfreq = max(1, dec_len // 2)
    return min(enc_nfreq, dec_nfreq)

def _clamp_fourier_modes_recursively(module, cap: int) -> None:
    """
    FEDformer 서브모듈 전체를 순회하며
    - modes1/modes2를 cap 이하로 줄이고
    - index_q/index_kv 길이도 modes 및 weights의 모드 차원과 cap에 맞춰 잘라 일치시킨다.
    """

    def _to_list(x):
        if x is None:
            return None
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        # torch.Tensor or numpy or other iterables
        try:
            return [int(v) for v in list(x)]
        except Exception:
            return None

    for m in module.modules():
        # 1) modes를 cap으로 제한
        if hasattr(m, "modes1"):
            try:
                m.modes1 = int(min(int(m.modes1), int(cap)))
            except Exception:
                pass
        if hasattr(m, "modes2"):
            try:
                m.modes2 = int(min(int(m.modes2), int(cap)))
            except Exception:
                pass

        # 2) 해당 모듈이 가진 weight 파라미터의 모드 차원 확인
        w1_lim = None  # 보통 kv 쪽 모드 차원
        w2_lim = None  # 보통 q 쪽 모드 차원
        if hasattr(m, "weights1") and getattr(m, "weights1") is not None:
            try:
                w1_lim = int(getattr(m, "weights1").shape[-1])
            except Exception:
                w1_lim = None
        if hasattr(m, "weights2") and getattr(m, "weights2") is not None:
            try:
                w2_lim = int(getattr(m, "weights2").shape[-1])
            except Exception:
                w2_lim = None

        # 3) 인덱스 배열을 cap/modes/weights 한계 내로 자르는 헬퍼
        def _trim_indices(idx, max_len):
            idx = _to_list(idx)
            if idx is None:
                return None
            # 음수/초과 index 제거, cap 범위 내로
            idx = [int(v) for v in idx if 0 <= int(v) < int(cap)]
            # 길이를 모드 길이에 맞춤
            if max_len is not None and len(idx) > int(max_len):
                idx = idx[: int(max_len)]
            # 비어버리면 간단한 연속 인덱스로 대체(안정성)
            if (max_len is not None) and int(max_len) > 0 and len(idx) == 0:
                idx = list(range(int(max_len)))
            return idx

        # 4) index_q 정리 (쿼리 쪽은 보통 modes1, weights2 제한)
        if hasattr(m, "index_q"):
            target_q = int(getattr(m, "modes1", cap))
            if w2_lim is not None:
                target_q = min(target_q, int(w2_lim))
            m.index_q = _trim_indices(getattr(m, "index_q"), target_q)

        # 5) index_kv 정리 (키/값 쪽은 보통 modes2, weights1 제한)
        if hasattr(m, "index_kv"):
            target_kv = int(getattr(m, "modes2", cap))
            if w1_lim is not None:
                target_kv = min(target_kv, int(w1_lim))
            m.index_kv = _trim_indices(getattr(m, "index_kv"), target_kv)


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
    F = cfg.FEDformer

    # enc/dec가 허용하는 최대 푸리에 모드 수
    enc_nfreq = max(1, cfg.SEQ_LEN // 2)
    dec_len = int(cfg.LABEL_LEN + cfg.HORIZON)
    dec_nfreq = max(1, dec_len // 2)

    wanted_modes = int(getattr(F, "MODES", 32))
    safe_modes = max(1, min(wanted_modes, enc_nfreq, dec_nfreq))  # ✅ 안전 클램프

    return SimpleNamespace(
        # sequence lengths
        seq_len=cfg.SEQ_LEN, label_len=cfg.LABEL_LEN, pred_len=cfg.HORIZON,
        # I/O dims
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        # model size
        d_model=F.D_MODEL, n_heads=F.N_HEADS, d_ff=F.D_FF,
        e_layers=F.E_LAYERS, d_layers=F.D_LAYERS, dropout=F.DROPOUT,
        # embed/time
        embed=F.EMBED, freq=F.FREQ,
        # attention variants
        version=getattr(F, "VERSION", "Fourier"),
        mode_select=getattr(F, "MODE_SELECT", "random"),
        modes=safe_modes,                 # ✅ 여기 안전값 적용
        moving_avg=getattr(F, "MOVING_AVG", 25),
        factor=getattr(F, "FACTOR", 1),
        # extras
        L=getattr(F, "L", 1),
        base=getattr(F, "BASE", "legendre"),
        cross_activation=getattr(F, "CROSS_ACTIVATION", "tanh"),
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

        _clamp_fourier_modes_recursively(original_model, _safe_modes_cap(cfg))

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
        x_mark_enc = batch.get("x_mark_enc", batch.get("x_mark"))
        y_mark_dec = batch.get("y_mark_dec", batch.get("y_mark"))
        # --- dtype을 x_enc와 강제로 맞춰 안전성 확보 ---
        if x_mark_enc.dtype != x_enc.dtype:
            x_mark_enc = x_mark_enc.to(x_enc.dtype)
        if y_mark_dec.dtype != x_enc.dtype:
            y_mark_dec = y_mark_dec.to(x_enc.dtype)

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