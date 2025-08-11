from __future__ import annotations
import optuna
from typing import Dict, Any, List


def _divisors(n: int, candidates: List[int]) -> List[int]:
    return [c for c in candidates if n % c == 0]


def _safe_heads(d_model: int) -> List[int]:
    # FEDformer/Autoformer/PatchTST에서 shape 오류를 피하려고 보수적 선택
    return _divisors(d_model, [1, 2, 4, 8])


def suggest(trial: optuna.trial.Trial, model: str, base_cfg) -> Dict[str, Any]:
    """
    모델별 탐색 공간을 정의하고, config override 딕셔너리를 반환.
    - 상위 공통: BATCH_SIZE, LR, WEIGHT_DECAY, SEQ_LEN
    - 모델별: "<model>.<param>" 키 (소문자 섹션명). apply_overrides에서 대문자 섹션으로 매핑됨.
    - Optuna의 '동적 분포' 에러를 피하려고, d_model에 의존하는 분포는
      파라미터 이름에 d값을 태깅(@...)해서 '분포가 고정'되도록 구성.
    """
    model = model.lower()
    o: Dict[str, Any] = {
        "BATCH_SIZE": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "LR": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "WEIGHT_DECAY": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "SEQ_LEN": trial.suggest_categorical("seq_len", [28, 42, 56, 84]),
    }

    # ===== 모델별 =====
    if model == "fedformer":
        d = trial.suggest_categorical("fedformer.d_model", [64, 128, 256, 384])
        h = trial.suggest_categorical(f"fedformer.n_heads@{d}#r2", _safe_heads(d))
        d_ff = trial.suggest_categorical(f"fedformer.d_ff@{d}#r2", [2 * d, 4 * d, 8 * d])

        # ✅ 여기: 고정 superset에서 뽑고, 파라미터 이름은 새로!
        raw_modes = trial.suggest_categorical("fedformer.modes_all", [8, 16, 32, 64, 128])

        o["fedformer.d_model"] = d
        o["fedformer.n_heads"] = h
        o["fedformer.d_ff"] = d_ff
        o["fedformer.e_layers"] = trial.suggest_int("fedformer.e_layers", 2, 6)
        o["fedformer.d_layers"] = trial.suggest_int("fedformer.d_layers", 1, 4)
        o["fedformer.dropout"] = trial.suggest_float("fedformer.dropout", 0.05, 0.35)

        # ✅ overrides에는 기존 키로 넣되, 값은 raw_modes (최종 클램프는 모델 쪽에서 수행)
        o["fedformer.modes"] = raw_modes

        o["fedformer.moving_avg"] = trial.suggest_categorical("fedformer.moving_avg", [13, 25, 49])
        o["fedformer.factor"] = trial.suggest_categorical("fedformer.factor", [1, 2, 3])
        return o

    if model == "autoformer":
        d = trial.suggest_categorical("autoformer.d_model", [64, 128, 256, 384])
        h = trial.suggest_categorical(f"autoformer.n_heads@{d}#r2", _safe_heads(d))
        d_ff = trial.suggest_categorical(f"autoformer.d_ff@{d}#r2", [2*d, 4*d, 8*d])

        o["autoformer.d_model"]    = d
        o["autoformer.n_heads"]    = h
        o["autoformer.d_ff"]       = d_ff
        o["autoformer.e_layers"]   = trial.suggest_int("autoformer.e_layers", 2, 6)
        o["autoformer.d_layers"]   = trial.suggest_int("autoformer.d_layers", 1, 4)
        o["autoformer.dropout"]    = trial.suggest_float("autoformer.dropout", 0.05, 0.35)
        o["autoformer.moving_avg"] = trial.suggest_categorical("autoformer.moving_avg", [13, 25, 49])
        o["autoformer.factor"]     = trial.suggest_categorical("autoformer.factor", [1, 2, 3])
        return o

    if model == "patchtst":
        d = trial.suggest_categorical("patchtst.d_model", [64, 128, 256])
        h = trial.suggest_categorical(f"patchtst.n_heads@{d}#r2", _safe_heads(d))
        ff = trial.suggest_categorical(f"patchtst.ff_dim@{d}#r2", [2*d, 4*d])

        patch_len = trial.suggest_categorical("patchtst.patch_len", [4, 8, 16, 24])
        stride_candidates = [1, 2, 4, 8]
        stride_candidates = [s for s in stride_candidates if s <= patch_len]
        stride = trial.suggest_categorical(f"patchtst.stride@{patch_len}", stride_candidates)

        o["patchtst.d_model"]  = d
        o["patchtst.n_heads"]  = h
        o["patchtst.ff_dim"]   = ff
        o["patchtst.depth"]    = trial.suggest_int("patchtst.depth", 2, 6)
        o["patchtst.dropout"]  = trial.suggest_float("patchtst.dropout", 0.0, 0.3)
        o["patchtst.patch_len"]= patch_len
        o["patchtst.stride"]   = stride
        return o

    return o
