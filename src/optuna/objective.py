from __future__ import annotations
import os
import gc
import optuna
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class _PruningCallback(pl.callbacks.Callback):
    """Lightning 1.x에서 안전하게 동작하는 Optuna 프루닝 콜백."""
    def __init__(self, trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        try:
            val = float(metric.detach().cpu().item())
        except Exception:
            val = float(metric)
        # 현재 epoch 기준으로 보고
        self.trial.report(val, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {trainer.current_epoch}, {self.monitor}={val:.4f}")

    # Lightning이 stateful 콜백을 점검할 때 에러 안 나게 비워둡니다.
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
from src.config import Config
from src.core.utils import seed_everything
from src.core.feature_engineer import FeatureEngineer
from src.core.data_module import TimeSeriesDataModule
from src.models.factory import build_model
from src.core.lightning_module import LitModel

from src.optuna.spaces import suggest
from src.optuna.utils import apply_overrides, rough_vram_score, cleanup_cuda


def _trial_ckpt_dir(model_name: str, trial: optuna.trial.Trial) -> str:
    d = os.path.join("results", "optuna", model_name, f"trial_{trial.number:04d}", "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def _warmup_forward_once(lit: LitModel, dm: TimeSeriesDataModule, raw_model) -> None:
    """학습 전에 한 배치로 shape/type 문제를 조기 검출."""
    # fit 단계 셋업 이후 호출되어야 함
    vloader = dm.val_dataloader()
    if vloader is None:
        return
    try:
        batch = next(iter(vloader))
    except StopIteration:
        return
    # Lightning에서 쓰는 동일 포맷으로 변환
    batch_dict = lit._format_batch(batch)
    raw_model.eval()
    with torch.no_grad():
        _ = raw_model(batch_dict)  # 오류 나면 여기서 즉시 raise


def make_objective(model_name: str):
    """Optuna objective 생성기: 모델명에 따라 search space를 구성하고 trial을 실행합니다."""
    def objective(trial: optuna.trial.Trial) -> float:
        base = Config()
        overrides = suggest(trial, model_name, base)
        cfg = apply_overrides(base, overrides)

        # Rough VRAM guard (모델별 섹션 인식)
        section = getattr(cfg, model_name.upper(), None)
        d_model = 128
        n_heads = 8
        if section is not None:
            d_model = getattr(section, "D_MODEL", d_model)
            # LSTM 등 head 개념이 없는 모델은 1로 둔다
            default_heads = 1 if model_name.lower() in {"lstm"} else 8
            n_heads = getattr(section, "N_HEADS", default_heads)

        if rough_vram_score(cfg.BATCH_SIZE, cfg.SEQ_LEN, d_model, n_heads) > 2_000_000:
            raise optuna.TrialPruned("Estimated VRAM too high")

        # 재현성 & 텐서코어 최적화
        seed_everything(cfg.SEED)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # 데이터 모듈
        fe = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
        num_workers = max(0, (os.cpu_count() or 4) // 2)

        dm = TimeSeriesDataModule(
            file_path=cfg.TRAIN_FILE,
            sequence_length=cfg.SEQ_LEN,
            forecast_horizon=cfg.HORIZON,
            label_len=cfg.LABEL_LEN,
            batch_size=cfg.BATCH_SIZE,
            feature_engineer=fe,
            num_workers=num_workers,
        )
        dm.prepare_data()
        dm.setup("fit")

        # 모델 & Lightning 모듈
        raw_model = build_model(model_name, cfg, input_dim=dm.input_dim)
        lit = LitModel(model=raw_model, cfg=cfg, item_names=list(dm.target_columns))

        # 학습 전에 1회 forward로 형상/타입 오류 즉시 검출
        _warmup_forward_once(lit, dm, raw_model)

        # 콜백들(체크포인트는 trial 전용 폴더에 best만 저장)
        es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.PATIENCE, verbose=False)
        ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=0)
        pr_cb = _PruningCallback(trial, monitor="val_loss")  # ← 우리 콜백 사용
        # 트레이너
        trainer = pl.Trainer(
            max_epochs=cfg.EPOCHS,
            accelerator=cfg.ACCELERATOR,  # 'auto' 권장
            devices=cfg.DEVICES,          # 'auto' 또는 [0]
            precision=cfg.PRECISION,      # ex) '16-mixed'
            callbacks=[es_cb, ckpt_cb, pr_cb],
            enable_progress_bar=False,
            log_every_n_steps=5,
        )

        try:
            trainer.fit(lit, dm)
            val = trainer.callback_metrics.get("val_loss")
            return float(val.detach().cpu().item()) if val is not None else float("inf")
        except RuntimeError as e:
            # CUDA OOM은 실패가 아니라 프루닝으로 간주
            if "out of memory" in str(e).lower():
                raise optuna.TrialPruned("CUDA OOM")
            raise
        finally:
            del trainer, lit, raw_model, dm
            cleanup_cuda()

    return objective
