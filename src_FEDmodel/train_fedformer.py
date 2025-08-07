# train.py
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import Config
from data_module import TimeSeriesDataModule, FeatureEngineer
from lightning_module import LitModel
sys.path.append('./models/FEDformer')
from models.fedformer import Model as FEDformer
from helpers import seed_everything

def train():
    cfg = Config()
    seed_everything(cfg.SEED)

    # 1. 데이터 모듈 준비
    feature_engineer = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(
        file_path=cfg.FILE_PATH,
        sequence_length=cfg.SEQ_LEN,
        forecast_horizon=cfg.HORIZON,
        label_len=cfg.LABEL_LEN,
        batch_size=cfg.BATCH_SIZE,
        feature_engineer=feature_engineer
    )
    dm.setup('fit')

    # 2. 모델 준비 (cfg의 모델 파라미터를 사용)
    # model_cfg = cfg.FEDformer
    # fedformer_model = FEDformer(
    #     enc_in=dm.output_dim,
    #     dec_in=dm.output_dim,
    #     c_out=dm.output_dim,
    #     seq_len=cfg.SEQ_LEN,
    #     label_len=cfg.LABEL_LEN,
    #     pred_len=cfg.HORIZON,
    #     d_model=model_cfg.D_MODEL,
    #     n_heads=model_cfg.N_HEADS,
    #     e_layers=model_cfg.E_LAYERS,
    #     d_layers=model_cfg.D_LAYERS,
    #     d_ff=model_cfg.D_FF,
    #     dropout=model_cfg.DROPOUT,
    #     output_attention=model_cfg.OUTPUT_ATTENTION,
    #     embed=model_cfg.EMBED,
    #     freq=model_cfg.FREQ,
    #     activation=model_cfg.ACTIVATION,
    #     version=model_cfg.VERSION,
    #     mode_select=model_cfg.MODE_SELECT,
    #     modes=model_cfg.MODES,
    #     moving_avg=model_cfg.MOVING_AVG,
    #     distil=model_cfg.DISTIL,
    #     factor=model_cfg.FACTOR,
    # )

    # [수정] FEDformer 모델이 요구하는 단일 config 객체를 생성합니다.
    class ModelConfigs:
        pass

    model_params = ModelConfigs()
    model_cfg = cfg.FEDformer # config.py의 FEDformer 중첩 클래스

    # 필요한 모든 파라미터를 model_params 객체의 속성으로 추가합니다.
    model_params.enc_in = dm.output_dim
    model_params.dec_in = dm.output_dim
    model_params.c_out = dm.output_dim
    model_params.seq_len = cfg.SEQ_LEN
    model_params.label_len = cfg.LABEL_LEN
    model_params.pred_len = cfg.HORIZON
    model_params.d_model = model_cfg.D_MODEL
    model_params.n_heads = model_cfg.N_HEADS
    model_params.e_layers = model_cfg.E_LAYERS
    model_params.d_layers = model_cfg.D_LAYERS
    model_params.d_ff = model_cfg.D_FF
    model_params.dropout = model_cfg.DROPOUT
    model_params.output_attention = model_cfg.OUTPUT_ATTENTION
    model_params.embed = model_cfg.EMBED
    model_params.freq = model_cfg.FREQ
    model_params.activation = model_cfg.ACTIVATION
    model_params.version = model_cfg.VERSION
    model_params.mode_select = model_cfg.MODE_SELECT
    model_params.modes = model_cfg.MODES
    model_params.moving_avg = model_cfg.MOVING_AVG
    model_params.distil = model_cfg.DISTIL
    model_params.factor = model_cfg.FACTOR
    model_params.num_time_features = len(dm.time_feature_columns)

    # [수정] 이제 파라미터가 담긴 객체 하나만 전달합니다.
    fedformer_model = FEDformer(model_params)

    # 3. Lightning Module 준비
    lit_model = LitModel(model=fedformer_model, config=cfg)

    # 4. 콜백 및 트레이너 설정
    # val_loss를 모니터링하여 가장 좋은 모델을 저장하는 콜백
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="fedformer_model/",
        filename="best-fedformer-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1, # 가장 좋은 모델 1개만 저장
        mode="min",   # monitor하는 지표가 낮을수록 좋음
    )

    # val_loss가 3 epoch 동안 개선되지 않으면 훈련을 멈추는 콜백
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.PATIENCE,
        verbose=True,
        mode="min"
    )

    # 5. Trainer 객체 생성
    trainer = pl.Trainer(
        max_epochs=cfg.MAX_EPOCHS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=True,
        log_every_n_steps=1,  # 10 스텝마다 로그 기록
    )

    # 6. 훈련 시작 (기존에는 trainer가 정의되지 않아 오류 발생)
    print("--- FEDformer 모델 훈련을 시작합니다 ---")
    trainer.fit(model=lit_model, datamodule=dm)
    print("--- 훈련이 완료되었습니다 ---")

if __name__ == '__main__':
    train()