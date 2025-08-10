from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn

class TimesFMModel(nn.Module):
    """
    Google TimesFM: 공개 레포는 추론 API 중심.
    -> 학습 파이프라인과 호환을 위해 'forward'만 정의(학습은 비활성).
    """
    def __init__(self, impl, cfg: Any):
        super().__init__()
        self._impl = impl   # timesfm.TimesFm 객체 (predict 전용)
        self._horizon = cfg.HORIZON

    @classmethod
    def from_config(cls, cfg: Any, input_dim: int) -> "TimesFMModel":
        # 로컬 레포: models/timesfm/src/timesfm/timesfm_torch.py
        root = Path(__file__).resolve().parents[2] / "models" / "timesfm" / "src"
        sys.path.insert(0, str(root))
        try:
            from timesfm.timesfm_torch import TimesFm  # type: ignore
        except Exception as e:
            raise ImportError("[TimesFM] timesfm_torch.TimesFm import 실패. models/timesfm 배치를 확인하세요.") from e
        tfm = TimesFm(checkpoint=None, device="cpu")  # 공개 예시 기준; 필요 시 인자 보완
        return cls(tfm, cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 입력: x_enc (B,L,N), 출력: (B,H,N)
        x = batch["x_enc"].detach().cpu().numpy()
        B, L, N = x.shape
        # TimesFM API는 (batch, length) 단일 시계열 위주 → 채널별 반복 호출
        outs = []
        for b in range(B):
            cols = []
            for n in range(N):
                pred = self._impl.forecast([x[b, :, n]], horizon=self._horizon)[0]  # API에 맞게 수정 필요
                cols.append(torch.tensor(pred, dtype=batch["x_enc"].dtype))
            outs.append(torch.stack(cols, dim=-1))
        return torch.stack(outs, dim=0)

'''
⚠️ 주의: TimesFM은 실제로는 사전학습 체크포인트 필요·API가 버전에 따라 다릅니다.
위 래퍼는 스켈레톤이며, 로컬 timesfm 버전의 TimesFm.forecast() 시그니처에 맞춰 1–2줄만 조정하면 동작합니다(파인튜닝은 별도 작업).
'''