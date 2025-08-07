import sys
import os
import torch
import pytorch_lightning as pl
import glob
from config import Config
from data_module import TimeSeriesDataModule, FeatureEngineer
from lightning_module import LitModel
sys.path.append('./models/FEDformer')
from models.fedformer import Model as FEDformer
# helpers.py에서 분석/시각화 함수를 가져옵니다.
from helpers import analyze_predictions, visualize_results, visualize_all_windows_for_one_item, visualize_all_windows_for_one_item2

def evaluate():
    print("--- 모델 성능 평가 및 분석을 시작합니다 ---")
    cfg = Config()

    # 1. 훈련 때와 동일하게 데이터 모듈을 준비합니다. (테스트 데이터셋용)
    feature_engineer = FeatureEngineer(cfg.LAG_PERIODS, cfg.MA_WINDOWS)
    dm = TimeSeriesDataModule(
        file_path=cfg.FILE_PATH,
        sequence_length=cfg.SEQ_LEN,
        forecast_horizon=cfg.HORIZON,
        label_len=cfg.LABEL_LEN,
        batch_size=cfg.BATCH_SIZE,
        feature_engineer=feature_engineer
    )
    dm.setup('test') # 'test' 데이터로 설정

    # 2. 훈련된 최고의 모델 체크포인트 경로를 찾습니다.
    #    (파일 이름은 train.py의 ModelCheckpoint 설정과 일치해야 합니다)
    checkpoint_dir = "fedformer_model/"
    best_checkpoint_path = glob.glob(f"{checkpoint_dir}/best-fedformer-model*.ckpt")
    best_checkpoint_path = max(best_checkpoint_path, key=os.path.getmtime)
    result_path = best_checkpoint_path.replace('fedformer_model', 'results').replace('.ckpt', '')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if not best_checkpoint_path:
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_dir}")
    
    print(f"로드할 체크포인트: {best_checkpoint_path[0]}")

    # 3. 체크포인트로부터 LitModel을 로드합니다.
    #    (모델 구조는 미리 전달해주어야 합니다)
    #    train.py의 모델 생성 코드와 동일한 로직입니다.
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
    
    with torch.serialization.safe_globals([Config]):
        lit_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path,
            model=fedformer_model,
            config=cfg # __init__에 config가 필요하므로 전달
        )

    # 4. Trainer를 생성하고 test() 메서드를 실행합니다.
    trainer = pl.Trainer(accelerator=cfg.ACCELERATOR, devices=cfg.DEVICES)
    trainer.test(model=lit_model, datamodule=dm)

    # 5. 모델 내부에 저장된 예측값과 실제값을 가져옵니다.
    predictions_tensor = lit_model.test_predictions
    actuals_tensor = lit_model.test_actuals

    # 6. 텐서를 Numpy 배열로 변환합니다.
    predictions_np = predictions_tensor.cpu().numpy()
    actuals_np = actuals_tensor.cpu().numpy()
    
    # 7. target_columns 정보를 DataModule에서 가져옵니다.
    target_columns = dm.target_columns

    # 8. analyze_predictions 함수를 호출합니다
    item_errors, horizon_errors = analyze_predictions(
        predictions=predictions_np,
        actuals=actuals_np,
        target_columns=target_columns
    )
    item_errors.to_csv(os.path.join(result_path, "item_errors.csv"), index=False, encoding='utf-8-sig')

    # 9. (선택) 시각화 함수도 호출할 수 있습니다.
    visualize_results(
        predictions=predictions_np,
        actuals=actuals_np,
        item_errors=item_errors,
        horizon_errors=horizon_errors,
        target_columns=target_columns,
        save_dir= result_path
    )

    visualize_all_windows_for_one_item(
        actuals=actuals_np,
        predictions=predictions_np,
        item_errors=item_errors,
        target_columns=target_columns,
        item_to_plot='worst', # 'best'로 변경하여 가장 잘 예측한 아이템을 볼 수도 있습니다.
        save_dir=result_path
    )

    visualize_all_windows_for_one_item(
        actuals=actuals_np,
        predictions=predictions_np,
        item_errors=item_errors,
        target_columns=target_columns,
        item_to_plot='best', # 'best'로 변경하여 가장 잘 예측한 아이템을 볼 수도 있습니다.
        save_dir=result_path
    )

    visualize_all_windows_for_one_item2(
        actuals=actuals_np,
        predictions=predictions_np,
        item_errors=item_errors,
        target_columns=target_columns,
        item_to_plot='worst', # 'best'로 변경하여 가장 잘 예측한 아이템을 볼 수도 있습니다.
        save_dir=result_path
    )

    visualize_all_windows_for_one_item2(
        actuals=actuals_np,
        predictions=predictions_np,
        item_errors=item_errors,
        target_columns=target_columns,
        item_to_plot='best', # 'best'로 변경하여 가장 잘 예측한 아이템을 볼 수도 있습니다.
        save_dir=result_path
    )

if __name__ == '__main__':
    evaluate()