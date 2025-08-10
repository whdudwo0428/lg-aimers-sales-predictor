from .model_fedformer import FedformerModel
from .model_autoformer import AutoformerModel
from .model_patchtst import PatchTSTModel
# 선택: TimesFM 추론용
try:
    from .model_timesfm import TimesFMModel
except Exception:
    TimesFMModel = None

REGISTRY = {
    "fedformer":  FedformerModel,
    "autoformer": AutoformerModel,
    "patchtst":   PatchTSTModel,
    "timesfm":    TimesFMModel,   # None이면 사용 시 ImportError 발생
}
def build_model(name: str, cfg, input_dim: int):
    name = (name or "fedformer").lower()
    if name not in REGISTRY or REGISTRY[name] is None:
        raise ImportError(f"[factory] Model '{name}' is not available.")
    return REGISTRY[name].from_config(cfg, input_dim)
