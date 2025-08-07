import torch

class Config:
    # --- 데이터, 공통 설정 ---
    SEED = 42
    FILE_PATH = 'dataset/train/train.csv'
    SEQ_LEN = 28
    HORIZON = 7
    LABEL_LEN = 28
    BATCH_SIZE = 64
    MAX_EPOCHS = 50
    PATIENCE = 10
    ACCELERATOR = 'auto'
    DEVICES = 'auto'

    # --- FEATURE ENGINEERING ---
    LAG_PERIODS = [7, 14]
    MA_WINDOWS = [7, 28]

    # --- 모델 공통 설정 ---
    LEARNING_RATE = 1e-4
    LOSS_FN = torch.nn.MSELoss()

    # --- FEDformer 모델 전용 하이퍼파라미터 ---
    class FEDformer:
        D_MODEL = 512
        N_HEADS = 8
        E_LAYERS = 2
        D_LAYERS = 1
        D_FF = 2048
        DROPOUT = 0.05
        OUTPUT_ATTENTION = True
        EMBED = 'timeF'
        FREQ = 'd'
        ACTIVATION = 'gelu'
        VERSION = 'Fourier'
        MODE_SELECT = 'random'
        MODES = 64
        MOVING_AVG = 25
        DISTIL = True
        FACTOR = 1