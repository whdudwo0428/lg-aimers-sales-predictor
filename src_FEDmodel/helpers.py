# 1. seed_everything: 재현성을 위해 랜덤 시드를 고정하는 함수
# 2. analyze_predictions: 예측 결과와 실제 값을 받아 성능 지표를 계산하는 분석 함수
# 3. visualize_results: 분석 결과를 시각화하여 이미지 파일로 저장하는 함수
# 4. convert_to_submission_format: 예측 결과를 최종 제출 형식으로 변환하는 함수

import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import platform

# 운영체제에 따라 다른 폰트 경로를 설정합니다.
if platform.system() == 'Darwin':  # Mac인 경우
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows인 경우
    plt.rc('font', family='Malgun Gothic')

# 한글 폰트 사용 시 마이너스 부호 깨짐 방지
plt.rc('axes', unicode_minus=False)

def seed_everything(seed: int = 42):
    """
    재현성을 위해 파이썬, Numpy, PyTorch의 랜덤 시드를 고정하는 함수입니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    대회 규정에 맞는 SMAPE를 계산합니다.
    - 실제 값이 0인 데이터 포인트는 계산에서 제외합니다.
    """
    # np.ndarray 형태로 변환
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # 실제 값이 0인 경우를 제외하기 위한 마스크 생성
    mask = actual != 0
    
    # 마스크를 적용하여 실제 값과 예측 값을 필터링
    actual_masked = actual[mask]
    predicted_masked = predicted[mask]
    
    # 만약 필터링 후 데이터가 없다면 (모든 실제 값이 0이었다면) 0을 반환
    if len(actual_masked) == 0:
        return 0.0
    
    # SMAPE 계산
    numerator = 2 * np.abs(predicted_masked - actual_masked)
    denominator = np.abs(actual_masked) + np.abs(predicted_masked)
    
    return np.mean(numerator / denominator)


def calculate_item_smapes(actuals: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """
    아이템별 SMAPE 점수를 계산합니다. (벡터화된 연산)
    - 실제 값이 0인 데이터 포인트는 계산에서 제외합니다.
    """
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    
    # 분자/분모 계산
    numerator = 2 * np.abs(predictions - actuals)
    denominator = np.abs(actuals) + np.abs(predictions)
    
    # 분모가 0인 경우(실제값과 예측값 모두 0)除算 오류를 피하기 위해 초기화
    smape_scores = np.zeros_like(actuals)
    
    # 분모가 0이 아닌 경우에만 SMAPE 값을 계산
    mask = denominator != 0
    smape_scores[mask] = numerator[mask] / denominator[mask]
    
    # 실제 값이 0인 경우는 평가에서 제외해야 하므로, 해당 위치의 점수를 NaN으로 만듭니다.
    smape_scores[actuals == 0] = np.nan
    
    # 각 아이템(axis=2)에 대해 샘플(axis=0)과 시점(axis=1)의 평균을 계산합니다.
    # np.nanmean은 NaN 값을 무시하고 평균을 계산합니다.
    item_smapes = np.nanmean(smape_scores, axis=(0, 1))
    
    # 만약 어떤 아이템의 실제값이 모두 0이어서 최종 점수가 NaN이 되면, 0으로 처리합니다.
    return np.nan_to_num(item_smapes, nan=0.0)


def analyze_predictions(predictions: np.ndarray, actuals: np.ndarray, target_columns: pd.Index):
    """
    예측 결과를 다각도로 분석하여 성능 지표를 계산하고 데이터프레임으로 반환합니다.
    """
    print("\\n--- 예측 결과 심층 분석 시작 ---")
    
    # 전체 성능 지표
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    smape_score = smape(actuals, predictions)
    print(f"Overall Test MAE: {mae:.4f}")
    print(f"Overall Test RMSE: {rmse:.4f}")
    print(f"Overall Test SMAPE: {smape_score:.4f}")
    
    # 1. 아이템별 성능 분석
    item_maes = np.mean(np.abs(predictions - actuals), axis=(0, 1))
    item_smape_scores = calculate_item_smapes(actuals, predictions)

    item_errors = pd.DataFrame({
        'item_name': target_columns,
        'mae': item_maes,
        'smape': item_smape_scores
    }).sort_values('smape', ascending=True)
    
    print("\\n[Best Predicted Items (Top 5)]")
    print(item_errors.head())
    print("\\n[Worst Predicted Items (Top 5)]")
    print(item_errors.tail())
    
    # 2. 예측 시점(Horizon)별 성능 분석
    horizon_maes = np.mean(np.abs(predictions - actuals), axis=(0, 2))
    horizon_errors = pd.DataFrame({
        'day': [f'Day {i+1}' for i in range(actuals.shape[1])],
        'mae': horizon_maes
    })
    
    print("\\n[MAE by Forecast Horizon]")
    print(horizon_errors)
    print("--- 분석 완료 ---")
    return item_errors, horizon_errors


def visualize_results(predictions: np.ndarray, actuals: np.ndarray, item_errors: pd.DataFrame, 
                      horizon_errors: pd.DataFrame, target_columns: pd.Index, save_dir: str = './results'):
    """
    분석 결과를 바탕으로 다양한 그래프를 생성하여 지정된 디렉토리에 저장합니다.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    print(f"\\n--- 분석 결과 시각화 시작 (결과는 '{save_dir}' 폴더에 저장됩니다) ---")
    
    # 1. Best/Worst 예측 아이템 시각화
    best_item_name = item_errors.iloc[0]['item_name']
    worst_item_name = item_errors.iloc[-1]['item_name']
    best_item_idx = target_columns.get_loc(best_item_name)
    worst_item_idx = target_columns.get_loc(worst_item_name)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(actuals[0, :, best_item_idx], label='Actual', marker='o')
    axes[0].plot(predictions[0, :, best_item_idx], label='Predicted', marker='x')
    axes[0].set_title(f"Best Predicted Item: {best_item_name}")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(actuals[0, :, worst_item_idx], label='Actual', marker='o')
    axes[1].plot(predictions[0, :, worst_item_idx], label='Predicted', marker='x')
    axes[1].set_title(f"Worst Predicted Item: {worst_item_name}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_worst_predictions.png"))
    plt.close() # Figure 객체를 닫아 메모리를 관리합니다.
    
    # 2. 예측 시점별 MAE 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(horizon_errors['day'], horizon_errors['mae'])
    plt.title("MAE by Forecast Horizon")
    plt.ylabel("Mean Absolute Error")
    plt.savefig(os.path.join(save_dir, "mae_by_horizon.png"))
    plt.close()

    # 3. 오차 분포 시각화
    residuals = (actuals - predictions).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Distribution of Prediction Errors (Actual - Predicted)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "error_distribution.png"))
    plt.close()

    print("--- 시각화 완료 ---")


def visualize_all_windows_for_one_item(
    actuals: np.ndarray,
    predictions: np.ndarray,
    item_errors: pd.DataFrame,
    target_columns: pd.Index,
    item_to_plot: str = 'worst', # 'worst', 'best', 또는 특정 아이템 이름
    save_dir: str = './results'
):
    """
    특정 아이템 하나에 대한 모든 슬라이딩 윈도우 예측을
    하나의 긴 이미지 파일로 저장합니다.
    """
    import os
    print(f"\\n--- '{item_to_plot}' 아이템 전체 예측 과정 시각화 시작 ---")

    # 분석할 아이템 선택
    if item_to_plot == 'worst':
        item_name = item_errors.iloc[-1]['item_name']
    elif item_to_plot == 'best':
        item_name = item_errors.iloc[0]['item_name']
    else:
        item_name = item_to_plot
    
    item_idx = target_columns.get_loc(item_name)
    
    num_windows = predictions.shape[0]
    
    # 전체 윈도우 개수만큼 세로로 긴 subplot을 생성합니다.
    fig, axes = plt.subplots(num_windows, 1, figsize=(12, 5 * num_windows))

    for i in range(num_windows):
        ax = axes[i]
        # i번째 윈도우의 실제값과 예측값을 그립니다.
        ax.plot(actuals[i, :, item_idx], label='Actual', marker='o')
        ax.plot(predictions[i, :, item_idx], label='Predicted', marker='x')
        ax.set_title(f"Forecast Window #{i+1} (Item: {item_name})")
        ax.legend()
        ax.grid(True)
        ax.set_ylabel('Sales Quantity')

    axes[-1].set_xlabel('Forecast Horizon (Day 1 to 7)') # 마지막 그래프에만 x축 라벨 추가
    
    plt.tight_layout()
    output_filename = f"all_windows_{item_to_plot}_item.png"
    output_path = os.path.join(save_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ 전체 예측 과정 그래프가 '{output_path}'에 저장되었습니다.")

def visualize_all_windows_for_one_item2(
    actuals: np.ndarray,
    predictions: np.ndarray,
    item_errors: pd.DataFrame,
    target_columns: pd.Index,
    item_to_plot: str = 'worst',
    save_dir: str = './results'
):
    """
    특정 아이템 하나에 대한 모든 슬라이딩 윈도우 예측을
    하나의 긴 이미지 파일로 저장합니다.
    'best'와 'worst' 선택 로직을 개선했습니다.
    """
    import os
    print(f"\\n--- '{item_to_plot}' 아이템 전체 예측 과정 시각화 시작 ---")

    # [수정] SMAPE가 0과 2인 극단적인 경우를 제외하고 의미있는 품목만 필터링
    filtered_errors = item_errors[(item_errors['smape'] > 0) & (item_errors['smape'] < 2)]
    
    if filtered_errors.empty:
        print(f"경고: SMAPE가 0과 2 사이인 분석 대상 품목이 없어 시각화를 건너뜁니다.")
        return

    # 필터링된 데이터프레임 내에서 Best/Worst를 선택합니다.
    if item_to_plot == 'worst':
        # SMAPE가 가장 높은 (가장 성능이 나쁜) 아이템 선택
        selected_item_name = filtered_errors.sort_values('smape', ascending=False).iloc[0]['item_name']
    elif item_to_plot == 'best':
        # SMAPE가 가장 낮은 (가장 성능이 좋은) 아이템 선택
        selected_item_name = filtered_errors.sort_values('smape', ascending=True).iloc[0]['item_name']
    else:
        # 특정 아이템 이름이 주어지면 해당 아이템을 사용
        if item_to_plot not in item_errors['item_name'].values:
            print(f"경고: '{item_to_plot}' 품목을 찾을 수 없습니다.")
            return
        selected_item_name = item_to_plot
    
    item_idx = target_columns.get_loc(selected_item_name)
    
    num_windows = predictions.shape[0]
    
    fig, axes = plt.subplots(num_windows, 1, figsize=(12, 5 * num_windows))
    
    if num_windows == 1: # 윈도우가 1개일 경우 axes가 배열이 아니므로 처리
        axes = [axes]

    for i in range(num_windows):
        ax = axes[i]
        ax.plot(actuals[i, :, item_idx], label='Actual', marker='o')
        ax.plot(predictions[i, :, item_idx], label='Predicted', marker='x')
        ax.set_title(f"Forecast Window #{i+1} (Item: {selected_item_name})")
        ax.legend()
        ax.grid(True)
        ax.set_ylabel('Sales Quantity')

    axes[-1].set_xlabel('Forecast Horizon (Day 1 to 7)')
    
    plt.tight_layout()
    output_filename = f"all_windows_{item_to_plot}_item_filtered2.png"
    output_path = os.path.join(save_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ '{selected_item_name}' 품목의 전체 예측 과정 그래프가 '{output_path}'에 저장되었습니다.")


def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame, id_to_date_map: dict):
    """
    예측 결과 데이터프레임을 최종 제출 형식으로 변환하는 함수입니다.
    (기존 코드에서 id_to_date_map을 인자로 받도록 수정하여 재사용성을 높였습니다.)
    """
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
        pred_df['매출수량']
    ))
    final_df = sample_submission.copy()
    menu_columns = final_df.columns[1:]
    final_df[menu_columns] = final_df[menu_columns].astype(float)

    for row_idx in final_df.index:
        placeholder_id = final_df.loc[row_idx, '영업일자']
        actual_date = id_to_date_map.get(placeholder_id, placeholder_id)

        if actual_date is None:
            continue

        for col in menu_columns:
            final_df.loc[row_idx, col] = pred_dict.get((actual_date, col), 0)
    
    final_df[menu_columns] = final_df[menu_columns].astype(int)
    return final_df