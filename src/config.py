"""
Global configuration for the forecasting pipeline.

This module centralises hyperparameters, file paths and runtime options
so different environments (macOS MPS / CUDA / CPU) can run with the same code.
"""

from __future__ import annotations
import os


class Config:
    # ---------------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------------
    MODEL_NAME: str = "fedformer"  # "fedformer" | "autoformer" | "patchtst" | "mamba"

    # ---------------------------------------------------------------
    # General / Paths
    # ---------------------------------------------------------------
    SEED: int = 42

    # Data paths (relative to project root when running: `python -m src.train_fedformer`)
    TRAIN_FILE: str = os.path.join("dataset", "train", "train.csv")
    TEST_DIR: str = os.path.join("dataset", "test")
    SAMPLE_SUBMISSION: str = os.path.join("dataset", "sample_submission.csv")

    RESULTS_DIR: str = "results"
    CHECKPOINT_DIR: str = os.path.join(RESULTS_DIR, "checkpoints")
    PREDS_DIR: str = os.path.join(RESULTS_DIR, "preds")
    SUBMISSION_FILE: str = os.path.join(RESULTS_DIR, "submission.csv")

    # Backward compatibility alias (some old scripts expect FILE_PATH)
    FILE_PATH: str = TRAIN_FILE

    # ---------------------------------------------------------------
    # Data handling
    # ---------------------------------------------------------------
    SEQ_LEN: int = 28
    HORIZON: int = 7
    LABEL_LEN: int = 28

    BATCH_SIZE: int = 32
    EPOCHS: int = 20

    # Use 0 on Windows if you hit DataLoader spawn issues
    NUM_WORKERS: int = 4

    # ---------------------------------------------------------------
    # Trainer runtime (cross-platform)
    # ---------------------------------------------------------------
    # Let Lightning auto-detect (CUDA/ROCm/MPS/CPU)
    ACCELERATOR: str = "auto"      # "auto" | "gpu" | "cpu" | "mps" | ...
    DEVICES: str | int | list[int] = "auto"  # "auto" or an int or list of device indices
    # Precision per PL 2.x naming: "32-true" | "16-mixed" | "bf16-mixed" | "64-true"
    # Keep a single source of truth—don't infer from AMP flag anymore.
    PRECISION: str = "32-true"

    # Logging & training niceties
    LOG_EVERY_N_STEPS: int = 5
    GRAD_CLIP: float = 0.0

    # ---------------------------------------------------------------
    # Optimisation
    # ---------------------------------------------------------------
    LR: float = 1e-3
    WD: float = 1e-4
    PATIENCE: int = 5

    # ---------------------------------------------------------------
    # Model: FEDformer
    # ---------------------------------------------------------------
    class FEDformer:
        D_MODEL: int = 128
        N_HEADS: int = 8
        E_LAYERS: int = 2
        D_LAYERS: int = 1
        D_FF: int = 512
        DROPOUT: float = 0.05

        OUTPUT_ATTENTION: bool = False
        EMBED: str = "timeF"
        FREQ: str = "d"
        ACTIVATION: str = "gelu"

        # Compatibility knobs (kept to match original repo arguments)
        VERSION: str = "Fourier"
        MODE_SELECT: str = "random"
        MODES: int = 32
        MOVING_AVG: int = 25
        DISTIL: bool = True
        FACTOR: int = 1

        # Wavelet variant (unused unless VERSION == 'Wavelets')
        L: int = 1
        BASE: str = "legendre"
        CROSS_ACTIVATION: str = "tanh"

    # ---------------------------------------------------------------
    # Feature engineering (date-only as per rules)
    # ---------------------------------------------------------------
    LAG_PERIODS: tuple[int, ...] = (7, 14)
    MA_WINDOWS: tuple[int, ...] = (7, 28)

    # ---------------------------------------------------------------
    # Placeholders for other models
    # ---------------------------------------------------------------
    class Autoformer:
        pass

    class PatchTST:
        pass

    class timesfm:
        pass
