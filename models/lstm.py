import os
import csv
from datetime import datetime
import random
import glob
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from tqdm import tqdm

# ----- 경로설정 및 데이터 로드 -----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(ROOT_DIR, "dataset", "train", "train.csv")
TEST_DIR = os.path.join(ROOT_DIR, "dataset", "test")
train = pd.read_csv(TRAIN_PATH)
# --- 고정 시드 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)



# --- 하이퍼파라미터 ---
LOOKBACK    = 28
PREDICT     = 7
BATCH_SIZE  = 16
EPOCHS      = 5
HIDDEN_DIM  = 64
NUM_LAYERS  = 2
LR          = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 모델 정의 ---
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=PREDICT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # (B, output_dim)


# --- 학습 ---
def train_lstm(train_df):
    trained_models = {}
    for (store_menu,), group in tqdm(train_df.groupby(['영업장명_메뉴명']), desc='Training LSTM'):
        store_train = group.sort_values('영업일자').copy()
        if len(store_train) < LOOKBACK + PREDICT:
            continue
        features = ['매출수량']
        scaler = MinMaxScaler()
        store_train[features] = scaler.fit_transform(store_train[features])
        train_vals = store_train[features].values
        x_train, y_train = [], []
        for i in range(len(train_vals) - LOOKBACK - PREDICT + 1):
            x_train.append(train_vals[i:i + LOOKBACK])
            y_train.append(train_vals[i + LOOKBACK:i + LOOKBACK + PREDICT, 0])
        x_train = torch.from_numpy(np.array(x_train)).float().to(DEVICE)
        y_train = torch.from_numpy(np.array(y_train)).float().to(DEVICE)
        model = MultiOutputLSTM(input_dim=1, output_dim=PREDICT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        model.train()
        for _ in range(EPOCHS):
            idx = torch.randperm(len(x_train))
            for i in range(0, len(x_train), BATCH_SIZE):
                batch_idx = idx[i:i + BATCH_SIZE]
                x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
                loss = criterion(model(x_batch), y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        trained_models[store_menu] = {
            'model': model.eval(),
            'scaler': scaler,
            'last_sequence': train_vals[-LOOKBACK:]
        }
    return trained_models


trained_models = train_lstm(train)


# --- 예측 ---
def predict_lstm(test_df, trained_models, test_prefix: str):
    results = []
    for (store_menu,), store_test in test_df.groupby(['영업장명_메뉴명']):
        key = store_menu
        if key not in trained_models:
            continue
        model = trained_models[key]['model']
        scaler = trained_models[key]['scaler']
        store_test_sorted = store_test.sort_values('영업일자')
        recent_vals = store_test_sorted['매출수량'].values[-LOOKBACK:]
        if len(recent_vals) < LOOKBACK:
            continue
        recent_vals_df = pd.DataFrame(recent_vals, columns=['매출수량'])
        recent_vals_scaled = scaler.transform(recent_vals_df)
        x_input = torch.from_numpy(recent_vals_scaled[np.newaxis, ...]).float().to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(x_input).squeeze().cpu().numpy()
        restored = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        restored = np.round(np.maximum(0, restored)).astype(int)
        pred_dates = [f"{test_prefix}+{i + 1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, restored):
            results.append({
                '영업일자': d,
                '영업장명_메뉴명': store_menu,
                '매출수량': val
            })
        # print(f"{store_menu} recent input:", recent_vals.flatten())
        # print(f"{store_menu} scaled input:", recent_vals_scaled.flatten())
        # print(f"{store_menu} prediction result:", pred_scaled)
    return pd.DataFrame(results)


# --- 전체 예측 실행 ---
all_preds = []
test_files = sorted(glob.glob(os.path.join(TEST_DIR, 'TEST_*.csv')))
for path in test_files:
    test_df = pd.read_csv(path)
    test_prefix = re.search(r'(TEST_\d+)', os.path.basename(path)).group(1)
    pred_df = predict_lstm(test_df, trained_models, test_prefix)
    all_preds.append(pred_df)

full_pred_df = pd.concat(all_preds, ignore_index=True)
print("\n전체 예측 완료! 예측 결과 shape:", full_pred_df.shape)

# wide format으로 변환 (baseline submission 형식)
sample_submission = pd.read_csv(os.path.join(ROOT_DIR, "dataset", "sample_submission.csv"))


def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
        pred_df['매출수량']
    ))
    final_df = sample_submission.copy()
    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)
    return final_df


submission = convert_to_submission_format(full_pred_df, sample_submission)

# --- 파일명 자동 구성 ---
MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]  # 예: lstm
PARAM_TAG = f"hidden{HIDDEN_DIM}_layers{NUM_LAYERS}_bs{BATCH_SIZE}_ep{EPOCHS}_lr{LR}"
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

submission_filename = f"submission_{MODEL_NAME}_{PARAM_TAG}.csv"
submission_path = os.path.join(OUTPUT_DIR, submission_filename)

# 저장
submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
print(f"\n제출용 CSV 저장 완료 → {submission_path}")

# --- summary 기록 ---
summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_entry = MODEL_NAME
params_entry = PARAM_TAG
score_entry = ""  # 나중에 validation/test 점수를 넣고 싶으면 여기에 대입

# 헤더가 없으면 새로 만들고, 있으면 이어붙이기
row = [timestamp, model_entry, params_entry, score_entry]
if not os.path.exists(summary_path):
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["타임스탬프", "모델명", "파라미터", "점수"])
        writer.writerow(row)
else:
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print(f"experiment_summary 업데이트됨 → {summary_path}")
