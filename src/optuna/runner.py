from __future__ import annotations
import argparse
import os
import json
import optuna
from optuna.pruners import MedianPruner

from src.optuna.objective import make_objective


def _ensure_sqlite_dir(storage: str) -> str:
    """
    sqlite:///... 형태일 때 DB 디렉터리를 미리 만들어줌.
    상대경로가 들어오면 현재 작업 디렉터리 기준 절대경로로 변환.
    """
    if not storage or not storage.startswith("sqlite://"):
        return storage
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        rest = storage[len(prefix):]
        db_path = rest.split("?", 1)[0]
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return prefix + db_path
    return storage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna runner")
    p.add_argument("--model", type=str, required=True, help="모델명 (예: fedformer, autoformer, patchtst, lstm 등)")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None, help="초 단위 제한시간")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None, help="예: sqlite:///$PWD/results/optuna/fedformer/study.sqlite3")
    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()

    # Storage 경로 보정(디렉터리 자동 생성)
    storage_url = _ensure_sqlite_dir(args.storage) if args.storage else None

    # Sampler/Pruner 설정
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,   # (Optuna 실험적) 상호작용 고려
        group=True,          # (Optuna 실험적) 같은 그룹 파라미터로 샘플
        n_startup_trials=10,
    )
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)

    # 스터디 생성/재사용
    study = optuna.create_study(
        storage=storage_url,
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # 동일 이름이면 이어서 진행
    )

    objective = make_objective(args.model)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=True)

    # 결과 요약 + best params 저장
    best = study.best_trial
    print(f"[Optuna] Best value: {best.value:.6f}")
    print("[Optuna] Best params:")
    for k, v in best.params.items():
        print(f"  - {k}: {v}")

    if storage_url and storage_url.startswith("sqlite:///"):
        out_dir = os.path.dirname(storage_url[len("sqlite:///"):])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best.params, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "best_value.txt"), "w", encoding="utf-8") as f:
            f.write(str(best.value))


if __name__ == "__main__":
    main()
