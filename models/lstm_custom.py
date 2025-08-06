import os, glob, re, math, random, csv
import datetime as _dt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ----------------------------
# 설정 및 경로
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(ROOT_DIR, "dataset", "train", "train.csv")
TEST_DIR = os.path.join(ROOT_DIR, "dataset", "test")
SUB_PATH = os.path.join(ROOT_DIR, "dataset", "sample_submission.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------
# 시드 & 디바이스
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 데이터 로드 & 전처리
# ----------------------------
train_df = pd.read_csv(TRAIN_PATH)
train_df["매출수량"] = train_df["매출수량"].clip(lower=0)
le_item = LabelEncoder().fit(train_df["영업장명_메뉴명"])
train_df["item_id"] = le_item.transform(train_df["영업장명_메뉴명"])
NUM_ITEMS = len(le_item.classes_)

# 가중치 (정규화)
WEIGHTS = {name: (2.0 if ("담하" in name or "미라시아" in name) else 1.0)
           for name in le_item.classes_}
weight_sum = sum(WEIGHTS.values())
WEIGHTS = {k: v / weight_sum * len(WEIGHTS) for k, v in WEIGHTS.items()}

# 제출 템플릿
sample_submission = pd.read_csv(SUB_PATH)

# ----------------------------
# 달력 피처
# ----------------------------
KOREAN_HOLIDAYS = {_dt.date(2023, 1, 1), _dt.date(2023, 1, 21), _dt.date(2023, 1, 22), _dt.date(2023, 1, 23),
                   _dt.date(2023, 1, 24), _dt.date(2023, 3, 1), _dt.date(2023, 5, 5), _dt.date(2023, 5, 27),
                   _dt.date(2023, 6, 6), _dt.date(2023, 8, 15), _dt.date(2023, 9, 28), _dt.date(2023, 9, 29),
                   _dt.date(2023, 9, 30), _dt.date(2023, 10, 2), _dt.date(2023, 10, 3), _dt.date(2023, 10, 9),
                   _dt.date(2023, 12, 25), _dt.date(2024, 1, 1), _dt.date(2024, 2, 9), _dt.date(2024, 2, 10),
                   _dt.date(2024, 2, 11), _dt.date(2024, 2, 12), _dt.date(2024, 3, 1), _dt.date(2024, 4, 10),
                   _dt.date(2024, 5, 5), _dt.date(2024, 5, 6), _dt.date(2024, 5, 15), _dt.date(2024, 6, 6),
                   _dt.date(2024, 8, 15), _dt.date(2024, 9, 16), _dt.date(2024, 9, 17), _dt.date(2024, 9, 18),
                   _dt.date(2024, 10, 3), _dt.date(2024, 10, 9), _dt.date(2024, 12, 25), _dt.date(2025, 1, 1),
                   _dt.date(2025, 1, 28), _dt.date(2025, 1, 29), _dt.date(2025, 1, 30), _dt.date(2025, 3, 3),
                   _dt.date(2025, 5, 5), _dt.date(2025, 5, 6), _dt.date(2025, 6, 6), _dt.date(2025, 8, 15),
                   _dt.date(2025, 10, 3), _dt.date(2025, 10, 6), _dt.date(2025, 10, 7), _dt.date(2025, 10, 8),
                   _dt.date(2025, 10, 9), _dt.date(2025, 12, 25)}

FEATURE_COLS = [
    "value", "dow_sin", "dow_cos", "dow2_sin", "dow2_cos",
    "mon_sin", "mon_cos", "doy_sin", "doy_cos", "is_wknd", "is_hol",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_7", "roll_var_7", "item_wd"
]
FEATURE_DIM = len(FEATURE_COLS)


def make_calendar_features(dates):
    ds = pd.to_datetime(dates)
    dow = ds.dt.dayofweek.values
    month = ds.dt.month.values
    doy = ds.dt.dayofyear.values
    is_wknd = (dow >= 5).astype(int)
    is_hol = ds.dt.date.isin(KOREAN_HOLIDAYS).astype(int)
    return np.stack([
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(4 * np.pi * dow / 7), np.cos(4 * np.pi * dow / 7),
        np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
        np.sin(2 * np.pi * (doy - 1) / 365), np.cos(2 * np.pi * (doy - 1) / 365),
        is_wknd, is_hol
    ], axis=1)


# ----------------------------
# 윈도우 생성
# ----------------------------
def get_windows(vals, dates, item_id, lb, horizon, stride):
    df = pd.DataFrame({"value": vals, "date": pd.to_datetime(dates)})

    # 달력 피처 생성
    cal_features = make_calendar_features(df["date"])
    df[["dow_sin", "dow_cos", "dow2_sin", "dow2_cos", "mon_sin", "mon_cos",
        "doy_sin", "doy_cos", "is_wknd", "is_hol"]] = cal_features

    # 지연 피처
    for lag in (1, 7, 14, 21, 28):
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # 롤링 피처
    df["roll_7"] = df["value"].shift(1).rolling(7, min_periods=1).mean()
    df["roll_var_7"] = df["value"].shift(1).rolling(7, min_periods=1).var().fillna(0)

    # 아이템-요일 상호작용
    df["weekday"] = df["date"].dt.dayofweek
    df["item_wd"] = item_id * 7 + df["weekday"]

    # NaN 제거
    df = df.dropna().reset_index(drop=True)

    if len(df) < lb + horizon:
        return np.array([]), np.array([])

    X, y = [], []
    for start in range(0, len(df) - lb - horizon + 1, stride):
        win = df.iloc[start:start + lb]
        tgt = df["value"].iloc[start + lb:start + lb + horizon].values
        X.append(win[FEATURE_COLS].values)
        y.append(tgt)
    return np.array(X), np.array(y)


# ----------------------------
# 손실 함수
# ----------------------------
class WeightedSMAPELoss(nn.Module):
    def __init__(self, weights_map, device):
        super().__init__()
        w = [weights_map[name] for name in le_item.classes_]
        # Fix: Move weight tensor to the specified device
        self.register_buffer("weight_tensor", torch.tensor(w, dtype=torch.float32, device=device))

    def forward(self, pred, true, item_ids):
        eps = 1e-6
        diff = torch.abs(pred - true)
        denom = torch.abs(pred) + torch.abs(true) + eps
        smape = 2 * diff / denom
        weights = self.weight_tensor[item_ids].unsqueeze(1)
        weighted_smape = smape * weights
        return weighted_smape.mean()


# ----------------------------
# 모델
# ----------------------------
class MultiOutputLSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, out_dim, dropout):
        super().__init__()
        self.pre_norm = nn.LayerNorm(FEATURE_DIM)
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            FEATURE_DIM, hid_dim, n_layers,
            batch_first=True, dropout=lstm_dropout
        )
        self.embedding = nn.Embedding(NUM_ITEMS, emb_dim)
        combined_dim = hid_dim + emb_dim
        self.post_norm = nn.LayerNorm(combined_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(combined_dim, out_dim)

    def forward(self, x, item_ids):
        x = self.pre_norm(x)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        emb = self.embedding(item_ids)
        combined = torch.cat([h, emb], dim=1)
        combined = self.post_norm(combined)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return F.relu(output)


# ----------------------------
# CV 평가
# ----------------------------
def evaluate_item_cv(vals, dates, item_id, cfg):
    X, y = get_windows(vals, dates, item_id,
                       cfg["lookback"], cfg["horizon"], cfg["stride"])

    if len(X) < 5:
        return None

    cv = TimeSeriesSplit(n_splits=3, test_size=cfg["horizon"])
    losses = []

    for tr, vl in cv.split(X):
        X_tr, y_tr = X[tr], y[tr]
        X_vl, y_vl = X[vl], y[vl]

        if len(X_tr) == 0 or len(X_vl) == 0:
            continue

        # 스케일링
        sx = StandardScaler()
        sy = StandardScaler()

        X_tr_scaled = X_tr.copy()
        X_vl_scaled = X_vl.copy()
        X_tr_scaled[:, :, 0] = sx.fit_transform(X_tr[:, :, 0].reshape(-1, 1)).reshape(-1, cfg["lookback"])
        X_vl_scaled[:, :, 0] = sx.transform(X_vl[:, :, 0].reshape(-1, 1)).reshape(-1, cfg["lookback"])

        y_tr_scaled = sy.fit_transform(y_tr)
        y_vl_scaled = sy.transform(y_vl)

        # 텐서 변환
        Xt = torch.tensor(X_tr_scaled, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y_tr_scaled, dtype=torch.float32, device=DEVICE)
        Xv = torch.tensor(X_vl_scaled, dtype=torch.float32, device=DEVICE)
        yv = torch.tensor(y_vl_scaled, dtype=torch.float32, device=DEVICE)

        idt = torch.full((len(Xt),), item_id, dtype=torch.long, device=DEVICE)
        idv = torch.full((len(Xv),), item_id, dtype=torch.long, device=DEVICE)

        # 모델 학습
        model = MultiOutputLSTM(cfg["emb_dim"], cfg["hidden_dim"],
                                cfg["num_layers"], cfg["horizon"],
                                cfg["dropout"]).to(DEVICE)

        opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        total_steps = cfg["epochs"] * math.ceil(len(Xt) / cfg["batch_size"])
        sched = OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total_steps,
                           pct_start=cfg["pct_start"], anneal_strategy=cfg["anneal_strategy"])

        # Fix: Pass DEVICE to the loss function
        crit = WeightedSMAPELoss(WEIGHTS, DEVICE)

        # 조기종료
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg["epochs"]):
            model.train()
            epoch_loss = 0
            num_batches = 0

            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(perm), cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                pred = model(Xt[idx], idt[idx])
                loss = crit(pred, yt[idx], idt[idx])

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches

            if avg_epoch_loss < best_val_loss - 1e-6:
                best_val_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        # 검증
        model.eval()
        with torch.no_grad():
            pred_v = model(Xv, idv)
            val_loss = crit(pred_v, yv, idv).item()

        losses.append(val_loss)

    return float(np.mean(losses)) if losses else None


# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial):
    cfg = {
        "lookback": trial.suggest_int("lookback", 21, 35),
        "horizon": 7,
        "stride": trial.suggest_int("stride", 3, 7),
        "emb_dim": trial.suggest_categorical("emb_dim", [16, 32, 64]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 5, 15),
        "patience": trial.suggest_int("patience", 3, 8),
        "pct_start": trial.suggest_float("pct_start", 0.1, 0.5),
        "anneal_strategy": trial.suggest_categorical("anneal_strategy", ["cos", "linear"])
    }

    scores = []
    items_processed = 0

    for name, grp in train_df.groupby("영업장명_메뉴명"):
        if items_processed >= 50:
            break

        score = evaluate_item_cv(
            grp["매출수량"].values,
            grp["영업일자"].values,
            le_item.transform([name])[0],
            cfg
        )

        if score is not None:
            scores.append(score)
            items_processed += 1

        if len(scores) > 0 and len(scores) % 10 == 0:
            current_score = np.mean(scores)
            trial.report(current_score, step=len(scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if not scores:
        raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))


# ----------------------------
# 예측 시퀀스 생성
# ----------------------------
def create_prediction_sequence(train_vals, train_dates, item_id, lookback):
    """훈련 데이터로부터 예측을 위한 시퀀스 생성"""
    df = pd.DataFrame({"value": train_vals, "date": pd.to_datetime(train_dates)})

    # 피처 생성
    cal_features = make_calendar_features(df["date"])
    df[["dow_sin", "dow_cos", "dow2_sin", "dow2_cos", "mon_sin", "mon_cos",
        "doy_sin", "doy_cos", "is_wknd", "is_hol"]] = cal_features

    for lag in (1, 7, 14, 21, 28):
        df[f"lag_{lag}"] = df["value"].shift(lag)

    df["roll_7"] = df["value"].shift(1).rolling(7, min_periods=1).mean()
    df["roll_var_7"] = df["value"].shift(1).rolling(7, min_periods=1).var().fillna(0)

    df["weekday"] = df["date"].dt.dayofweek
    df["item_wd"] = item_id * 7 + df["weekday"]

    df = df.dropna().reset_index(drop=True)

    if len(df) < lookback:
        return None

    last_sequence = df.iloc[-lookback:][FEATURE_COLS].values
    return last_sequence


# ----------------------------
# 학습 및 예측
# ----------------------------
def train_and_predict(best):
    cfg = best.params.copy()
    cfg["horizon"] = 7

    trained_models = {}

    print("모델 학습 중...")
    for name, grp in tqdm(train_df.groupby("영업장명_메뉴명"), desc="Training"):
        grp = grp.sort_values("영업일자")
        item_id = le_item.transform([name])[0]

        X, y = get_windows(grp["매출수량"].values, grp["영업일자"].values,
                           item_id, cfg["lookback"], cfg["horizon"], cfg["stride"])

        if len(X) == 0:
            continue

        # 스케일링
        sx, sy = StandardScaler(), StandardScaler()
        X_scaled = X.copy()
        X_scaled[:, :, 0] = sx.fit_transform(X[:, :, 0].reshape(-1, 1)).reshape(-1, cfg["lookback"])
        y_scaled = sy.fit_transform(y)

        # 텐서 변환
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y_scaled, dtype=torch.float32, device=DEVICE)
        ids = torch.full((len(Xt),), item_id, dtype=torch.long, device=DEVICE)

        # 모델 학습
        model = MultiOutputLSTM(cfg["emb_dim"], cfg["hidden_dim"],
                                cfg["num_layers"], cfg["horizon"],
                                cfg["dropout"]).to(DEVICE)

        opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps = math.ceil(len(Xt) / cfg["batch_size"])
        sched = OneCycleLR(opt, max_lr=cfg["lr"],
                           total_steps=cfg["epochs"] * steps,
                           pct_start=cfg["pct_start"],
                           anneal_strategy=cfg["anneal_strategy"])

        # Fix: Pass DEVICE to the loss function
        crit = WeightedSMAPELoss(WEIGHTS, DEVICE)

        for epoch in range(cfg["epochs"]):
            model.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(perm), cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                pred = model(Xt[idx], ids[idx])
                loss = crit(pred, yt[idx], ids[idx])

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

        trained_models[name] = {
            'model': model.eval(),
            'sx': sx,
            'sy': sy,
            'item_id': item_id
        }

    # 예측
    print("예측 중...")
    rows = []
    for path in sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv"))):
        df_test = pd.read_csv(path)
        prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)

        for name in df_test["영업장명_메뉴명"].unique():
            if name not in trained_models:
                continue

            model_info = trained_models[name]
            model = model_info['model']
            sx = model_info['sx']
            sy = model_info['sy']
            item_id = model_info['item_id']

            # 훈련 데이터에서 마지막 시퀀스 생성
            hist = train_df[train_df["영업장명_메뉴명"] == name].sort_values("영업일자")

            last_sequence = create_prediction_sequence(
                hist["매출수량"].values,
                hist["영업일자"].values,
                item_id,
                cfg["lookback"]
            )

            if last_sequence is None:
                continue

            # 스케일링
            scaled_sequence = last_sequence.copy()
            scaled_sequence[:, 0] = sx.transform(scaled_sequence[:, [0]]).flatten()

            # 예측
            X_input = torch.tensor(scaled_sequence[np.newaxis], dtype=torch.float32, device=DEVICE)
            item_input = torch.tensor([item_id], device=DEVICE)

            with torch.no_grad():
                pred_scaled = model(X_input, item_input).cpu().numpy().flatten()

            # 역변환
            pred_original = sy.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
            pred_final = np.round(np.clip(pred_original, 0, None)).astype(int)

            # 결과 저장
            for i, v in enumerate(pred_final):
                rows.append({
                    "영업일자": f"{prefix}+{i + 1}일",
                    "영업장명_메뉴명": name,
                    "매출수량": int(v)
                })

    pred_df = pd.DataFrame(rows, columns=["영업일자", "영업장명_메뉴명", "매출수량"])
    return pred_df, cfg


# ----------------------------
# 제출 형식 변환
# ----------------------------
def convert_to_submission_format(pred_df, template_df):
    d = dict(zip(
        zip(pred_df["영업일자"], pred_df["영업장명_메뉴명"]),
        pred_df["매출수량"]
    ))
    out = template_df.copy()
    for idx in out.index:
        date = out.at[idx, "영업일자"]
        for col in out.columns[1:]:
            out.at[idx, col] = d.get((date, col), 0)
    return out


def on_trial_complete(_s, t):
    print(f"[Trial {t.number}] SMAPE={t.value:.6f}, params={t.params}")


def main():
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(n_startup_trials=5, seed=42),
        pruner=MedianPruner(n_startup_trials=5, interval_steps=5),
        study_name="lstm_ts_cv", load_if_exists=True
    )
    study.optimize(objective, n_trials=30, timeout=3600, callbacks=[on_trial_complete])
    best = study.best_trial
    print(f"▶ Best SMAPE={best.value:.6f}, params={best.params}")

    preds, cfg = train_and_predict(best)
    submission = convert_to_submission_format(preds, sample_submission)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (f"lb{cfg['lookback']}_st{cfg['stride']}_eb{cfg['emb_dim']}"
           f"_hd{cfg['hidden_dim']}_nl{cfg['num_layers']}"
           f"_bs{cfg['batch_size']}_lr{cfg['lr']:.0e}"
           f"_wd{cfg['weight_decay']:.0e}")
    fname = f"submission_lstm_custom_{tag}_{ts}.csv"
    submission.to_csv(os.path.join(RESULTS_DIR, fname), index=False, encoding="utf-8-sig")
    print(f"✎ Saved → {fname}")

    summary = os.path.join(RESULTS_DIR, "experiment_summary.csv")
    row = [
        _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lstm_custom", tag, f"{best.value:.6f}"
    ]
    write_header = not os.path.exists(summary)
    with open(summary, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header: w.writerow(["Timestamp", "Model", "Params", "SMAPE"])
        w.writerow(row)
    print("✔ experiment_summary.csv updated")


if __name__ == "__main__":
    main()