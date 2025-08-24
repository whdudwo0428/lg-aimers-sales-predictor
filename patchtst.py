# ----------  # <CELL: imports & device>
import os, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from transformers import (
    PatchTSTConfig, PatchTSTForPrediction,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)
from tsfm_public.toolkit.dataset import ForecastDFDataset
from torch.utils.data import Subset
from transformers import TrainerCallback

print("CUDA available:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cuda.matmul.allow_tf32 = True
print("DEVICE:", DEVICE)

# ----------  # <CELL: global configs & paths>
CONTEXT_LEN = 28

CONTEXT_SET = [28, 56, 84]
def ctx_save_dir(ctx: int) -> str:
    d = os.path.join(SAVE_DIR, f"ctx_{ctx}")
    os.makedirs(d, exist_ok=True)
    return d

PRED_LEN    = 7
PATCH_LEN   = 7
PATCH_STRIDE= 1       # 7 / 1
DATA_STRIDE = 1

K_FOLDS = 5
PURGE_GAP_WEEKS = 1
ANCHOR_STEP = 7

SAVE_DIR = "./patchtst_sales_forecast"
os.makedirs(SAVE_DIR, exist_ok=True)
LE_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")

STORE_LE_PATH = os.path.join(SAVE_DIR, "store_label_encoder.pkl")

ROOT_CANDIDATES = ["./dataset", ".", "/mnt/data"]

def find_train_csv():
    for root in ROOT_CANDIDATES:
        for rel in ["train/train.csv", "train.csv"]:
            p = os.path.join(root, rel)
            if os.path.exists(p):
                return p
    raise FileNotFoundError("train.csv not found (tried ./dataset/train/train.csv, ./dataset/train.csv, /mnt/data/...).")

def find_test_files():
    # 우선 ./dataset/test/TEST_*.csv 찾고, 없으면 /mnt/data/TEST_*.csv
    for root in ROOT_CANDIDATES:
        pats = sorted(glob.glob(os.path.join(root, "test", "TEST_*.csv")))
        if pats:
            return pats
    pats = sorted(glob.glob("/mnt/data/TEST_*.csv"))
    return pats

CAP_MULT = 1.4                 # 상한 여유 배수
ENSEMBLE_NAIVE_W = 0.35  # 모델:(1-α)=0.50, 나이브:α=0.50  (권장 탐색 0.2~0.5)
FOLD_ENSEMBLE = True           # 폴드 앙상블 추론 활성화

# Loss 가중치(원-스케일 sMAPE 중심 + log-MAE 보강 + 0-overshoot 패널티)
SPLIT_OBJECTIVE = "SMAPE"   # 기존 LEADERBOARD_OBJECTIVE와 의미 동일
SMAPE_WEIGHT    = 0.85
MAE_WEIGHT      = 0.0       # zero-heavy 데이터면 원-MAE 비중은 낮추는 게 sMAPE에 유리
LOG_MAE_WEIGHT  = 0.12       # log-space 안정화(저수량/제로 근처 진동 억제)
SMAPE_EPS       = 1e-6      # sMAPE 분모 안정화용(원한다면 1e-5~1e-4로 상향 테스트)

# y_true==0일 때 양수 예측(overshoot)에 대한 별도 패널티(작게라도 양수 찍는 습성 억제)
ZERO_OVERSHOOT_PENALTY = 0.04   # λ_zero (0.15~0.5 권장 범위)

# EarlyStopping 공통 설정(이미 쓰셨다면 그대로 두셔도 됩니다)
EARLY_STOP_PATIENCE = 6  # CV/Final 모두 동일하게 사용

# 추론 단계(리더보드 직결) 안전장치
USE_INT_ROUND      = False   # 제출이 정수 필수 아니라고 하셨으므로 기본 False 권장
CUT_THRESHOLD      = None    # 이하면 0으로 컷(0.7~1.0 사이 탐색)
ZERO_RUN_GUARD_DAYS= 0      # 직전 K일 합이 0이면 미래 7일 전부 0 강제

# ----------  # <CELL: io & features>

def load_train_df():
    p = find_train_csv()
    print("Using train.csv:", p)
    df = pd.read_csv(p)
    # 기대 컬럼: 영업일자, 영업장명_메뉴명, 매출수량
    df["date"] = pd.to_datetime(df["영업일자"])
    df["sales"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0)
    df.loc[df["sales"] < 0, "sales"] = 0
    s = df["영업장명_메뉴명"].astype(str).str.split("_", n=1, expand=True)
    df["store_name"] = s[0]; df["menu_name"] = s[1]
    df["store_menu"] = df["store_name"] + "_" + df["menu_name"]
    return df

def add_rolling_channels(df: pd.DataFrame, group_col="store_menu") -> pd.DataFrame:
    df = df.sort_values([group_col,"date"]).copy()
    def _per_group(g):
        s = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0.0)
        # 미래 누수 방지: shift(1) 후 rolling
        g["roll7_mean"]   = s.shift(1).rolling(7,  min_periods=1).mean()
        g["roll28_mean"]  = s.shift(1).rolling(28, min_periods=1).mean()
        g["roll7_med"]    = s.shift(1).rolling(7,  min_periods=1).median()
        g["nzrate28"]     = (s.shift(1) > 0).astype(float).rolling(28, min_periods=1).mean()
        return g
    df = df.groupby(group_col, group_keys=False).apply(_per_group)
    for c in ["roll7_mean","roll28_mean","roll7_med","nzrate28"]:
        df[c] = df[c].fillna(0.0).astype(float)
    return df

def fit_or_load_label_encoder(series: pd.Series) -> LabelEncoder:
    if os.path.exists(LE_PATH):
        with open(LE_PATH, "rb") as f:
            le = pickle.load(f)
        new = sorted(set(series.astype(str)) - set(le.classes_))
        if new:
            le.classes_ = np.array(list(le.classes_) + list(new))
            # ★ 확장 시 즉시 저장
            with open(LE_PATH, "wb") as f:
                pickle.dump(le, f)
    else:
        le = LabelEncoder().fit(series.astype(str))
        with open(LE_PATH, "wb") as f:
            pickle.dump(le, f)
    return le

from holidays import country_holidays
from datetime import timedelta

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weekday"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["is_ski_season"] = df["month"].isin([12, 1, 2]).astype(int)

    years = sorted(df["date"].dt.year.unique().tolist())
    kr = set(country_holidays("KR", years=years))
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in kr)).astype(int)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["month_sin"]   = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"]   = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    # ---- 추가 캘린더 피처 ----
    df["day"]        = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    df["day_sin"]  = np.sin(2*np.pi*(df["day"]-1)/31.0)
    df["day_cos"]  = np.cos(2*np.pi*(df["day"]-1)/31.0)
    df["weekofyear_sin"] = np.sin(2*np.pi*(df["weekofyear"]-1)/53.0)
    df["weekofyear_cos"] = np.cos(2*np.pi*(df["weekofyear"]-1)/53.0)

    df["is_month_start"]   = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]     = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"]   = df["date"].dt.is_quarter_end.astype(int)

    # 휴일 전/후 ±1, ±2일 플래그 (위에서 만든 kr 그대로 사용)
    df["pre_holiday_1"]  = df["date"].dt.date.map(lambda d: int((d + timedelta(days=1)) in kr)).astype(int)
    df["post_holiday_1"] = df["date"].dt.date.map(lambda d: int((d - timedelta(days=1)) in kr)).astype(int)
    df["pre_holiday_2"]  = df["date"].dt.date.map(lambda d: int((d + timedelta(days=2)) in kr)).astype(int)
    df["post_holiday_2"] = df["date"].dt.date.map(lambda d: int((d - timedelta(days=2)) in kr)).astype(int)



    return df


def finalize_columns(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    out = df.copy()
    # --- 기존 sales/sales_log/정렬/주차 로직 그대로 ---
    if "sales" not in out.columns:
        if "매출수량" in out.columns:
            out["sales"] = pd.to_numeric(out["매출수량"], errors="coerce").fillna(0)
        else:
            out["sales"] = 0
    out.loc[out["sales"] < 0, "sales"] = 0
    out["sales_log"] = np.log1p(out["sales"])

    out["store_menu_id"] = le.transform(out["store_menu"].astype(str))

    # ADD ↓ 정적 카테고리로 쓸 store_id 생성
    store_le = fit_or_load_store_le(out["store_name"])
    out["store_id"] = store_le.transform(out["store_name"].astype(str))

    out = out.sort_values(["store_menu_id", "date"]).reset_index(drop=True)
    out["week_idx"] = ((out["date"] - out["date"].min()).dt.days // 7)
    return out

def build_item_caps_from_original():
    # 불연속 보강 전의 원본 분포 기반(양수만)으로 견고한 상한 계산
    orig = load_train_df()  # 원본 로드
    orig["date"] = pd.to_datetime(orig["영업일자"])
    orig["매출수량"] = pd.to_numeric(orig["매출수량"], errors="coerce").fillna(0)
    pos = orig[orig["매출수량"] > 0].copy()
    if pos.empty:
        return {}
    def robust_cap(g):
        a = g["매출수량"].to_numpy()
        q95 = np.quantile(a, 0.95)
        r = g.sort_values("date").tail(90)["매출수량"].to_numpy()
        r_q99 = np.quantile(r, 0.99) if r.size else q95
        return max(q95, r_q99)
    return pos.groupby("영업장명_메뉴명").apply(robust_cap).to_dict()

def enforce_regular_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    아이템(store_menu)별 관측 구간[min(date)..max(date)]을 D(일) 그리드로 강제.
    누락된 날은 sales=0 으로 보강 → 불연속 시퀀스 제거.
    """
    outs = []
    for key, g in df.groupby("store_menu", sort=False):
        g = g.sort_values("date")
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g2 = g.set_index("date").reindex(full_idx)
        g2.index.name = "date"

        # 식별자/문자열 컬럼 유지
        for c in ["store_name", "menu_name", "store_menu", "영업장명_메뉴명"]:
            if c in g2.columns:
                g2[c] = g[c].iloc[0]
        # 수치 보강
        if "sales" in g2.columns:
            g2["sales"] = pd.to_numeric(g2["sales"], errors="coerce").fillna(0)
        if "매출수량" in g2.columns:
            g2["매출수량"] = pd.to_numeric(g2["매출수량"], errors="coerce").fillna(0)

        outs.append(g2.reset_index())
    return pd.concat(outs, ignore_index=True)

def fit_or_load_store_le(series: pd.Series) -> LabelEncoder:
    if os.path.exists(STORE_LE_PATH):
        with open(STORE_LE_PATH, "rb") as f:
            le = pickle.load(f)
        new = sorted(set(series.astype(str)) - set(le.classes_))
        if new:
            le.classes_ = np.array(list(le.classes_) + list(new))
            with open(STORE_LE_PATH, "wb") as f:
                pickle.dump(le, f)
    else:
        le = LabelEncoder().fit(series.astype(str))
        with open(STORE_LE_PATH, "wb") as f:
            pickle.dump(le, f)
    return le

# ----------  # <CELL: helpers for inference>  (NEW)

def leftpad_to_context(g: pd.DataFrame, context_len: int, store_menu: str) -> pd.DataFrame:
    """
    단일 아이템 g(date 정렬된 DF)에 대해 길이가 context_len보다 짧으면
    왼쪽(과거)으로 제로패딩을 붙여 정확히 context_len을 맞춘다.
    공변량도 정상 생성되도록 add_time_features 호출.
    """
    g = g.sort_values("date").copy()
    n = len(g)
    if n >= context_len:
        return g

    need = context_len - n
    pad_end = g["date"].min() - pd.Timedelta(days=1)
    pad_dates = pd.date_range(end=pad_end, periods=need, freq="D")

    store, menu = store_menu.split("_", 1)
    pad = pd.DataFrame({
        "date": pad_dates,
        "영업일자": pad_dates,
        "store_name": store,
        "menu_name": menu,
        "store_menu": store_menu,
        "영업장명_메뉴명": store_menu,
        "매출수량": 0,
        "sales": 0,
    })
    pad = add_time_features(pad)
    g2 = pd.concat([pad, g], ignore_index=True)
    return g2

def _naive_last7(g: pd.DataFrame) -> np.ndarray:
    """최근 7일 평균을 7일로 복제하는 보수적 naive."""
    v = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0).to_numpy()
    if len(v) == 0:
        return np.zeros(PRED_LEN, dtype=float)
    tail = v[-7:] if len(v) >= 7 else v
    m = float(tail.mean())
    return np.full(PRED_LEN, m, dtype=float)

def _naive_same_dow(g: pd.DataFrame) -> np.ndarray:
    """최근 최대 4주(28일)에서 요일별 평균을 써서 7일 예측."""
    v = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0).to_numpy()
    if len(v) < 7:
        return np.zeros(PRED_LEN, dtype=float)
    n = min(28, len(v))
    tail = v[-n:]
    k = n // 7
    tail = tail[-(k*7):]  # 7의 배수로 맞춤
    if k == 0:
        return np.zeros(PRED_LEN, dtype=float)
    arr = tail.reshape(k, 7)
    mean_dow = arr.mean(axis=0)  # (7,)
    return mean_dow.astype(float)

def _blend_with_naive(yhat: np.ndarray, g: pd.DataFrame, alpha: float | None = None) -> np.ndarray:
    base_a = ENSEMBLE_NAIVE_W if alpha is None else float(alpha)

    v = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0.0).to_numpy()
    v28 = v[-28:] if v.size >= 28 else v
    nz28 = float((v28 > 0).mean()) if v28.size else 0.0  # 최근 28일 비제로율

    # 주간성 강도 (요일별 평균 분산 / 전체 분산)
    try:
        dow = g["weekday"].to_numpy()[-len(v28):]
        if v28.size >= 14 and np.var(v28) > 0:
            dow_means = [v28[dow == d].mean() for d in range(7) if (dow == d).any()]
            weekly_strength = float(np.var(dow_means) / (np.var(v28) + 1e-9)) if len(dow_means) >= 2 else 0.0
        else:
            weekly_strength = 0.0
    except Exception:
        weekly_strength = 0.0

    a = base_a * np.clip(weekly_strength, 0.0, 1.0) * (0.5 + 0.5 * nz28)
    a = float(np.clip(a, 0.0, base_a))  # 상한: base_a

    n1 = _naive_last7(g)
    n2 = _naive_same_dow(g)
    naive = 0.5 * n1 + 0.5 * n2
    return (1.0 - a) * yhat + a * naive

def _weekly_strength_and_nz(g: pd.DataFrame) -> tuple[float, float]:
    """요일성 강도(0~1 근사)와 최근 28일 비제로율"""
    v = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0.0).to_numpy()
    n = v.size
    v28 = v[-28:] if n >= 28 else v
    nz28 = float((v28 > 0).mean()) if v28.size else 0.0

    try:
        dow = g["weekday"].to_numpy()[-len(v28):]
        if v28.size >= 14 and np.var(v28) > 0:
            means = [v28[dow == d].mean() for d in range(7) if (dow == d).any()]
            weekly_strength = float(np.var(means) / (np.var(v28) + 1e-9)) if len(means) >= 2 else 0.0
        else:
            weekly_strength = 0.0
    except Exception:
        weekly_strength = 0.0

    return float(np.clip(weekly_strength, 0.0, 1.0)), nz28

def _context_mix_weights(g: pd.DataFrame, contexts: list[int]) -> dict[int, float]:
    """
    컨텍스트별 기본 가중 + (요일성/희소성) 보정.
    • 기본: {28:0.35, 56:0.40, 84:0.25} (없는 ctx는 자동 정규화)
    • 요일성↑ → 긴 컨텍스트(84/56)에 +, 짧은(28)에 -
    • 비제로율↓(희소) → 짧은(28)에 +, 긴(84)에 -
    """
    base = {28: 0.35, 56: 0.40, 84: 0.25}
    w = np.array([base.get(c, 0.0) for c in contexts], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(contexts), dtype=float) / len(contexts)

    weekly_strength, nz28 = _weekly_strength_and_nz(g)
    # 보정폭
    adj_w = np.zeros_like(w)
    for i, c in enumerate(contexts):
        # 요일성: 긴 컨텍스트에 가산, 짧은 컨텍스트에 감산
        adj_w[i] += ( 0.15 * weekly_strength) * (1 if c >= 56 else -1)
        # 희소 시계열(nz 낮음): 짧은 컨텍스트에 가산, 긴 컨텍스트에 감산
        adj_w[i] += ( 0.10 * (1.0 - nz28))   * (1 if c == 28 else (-1 if c >= 84 else 0))

    w = np.clip(w + adj_w, 1e-6, None)
    w = (w / w.sum()).astype(float)
    return {c: float(w[i]) for i, c in enumerate(contexts)}

def _zero_run_guard(g: pd.DataFrame, yhat: np.ndarray) -> np.ndarray:
    """최근 ZERO_RUN_GUARD_DAYS가 전부 0이면 미래 7일 0으로 가드."""
    v = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0).to_numpy()
    if ZERO_RUN_GUARD_DAYS > 0 and len(v) >= ZERO_RUN_GUARD_DAYS:
        if v[-ZERO_RUN_GUARD_DAYS:].sum() == 0:
            return np.zeros_like(yhat, dtype=float)
    return yhat

# ----------  # <CELL: dataset builders>
import inspect

DEBUG_DATASET_SIG = False

ID_COLS = ["store_menu_id"]
TIME_COL = "date"
TARGET_COLS = ["sales_log"]
# KNOWN_REAL_COLS 확장
KNOWN_REAL_COLS = [
    "is_holiday", "is_weekend", "is_ski_season",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "day_sin","day_cos","weekofyear_sin","weekofyear_cos",
    "is_month_start","is_month_end","is_quarter_start","is_quarter_end",
    "pre_holiday_1","post_holiday_1","pre_holiday_2","post_holiday_2",
    "roll7_mean","roll28_mean","roll7_med","nzrate28",
]

def build_dataset(
    df_split: pd.DataFrame,
    context_len: int | None = None,
    prediction_len: int | None = None,
    known_real_cols: list | None = None,
) -> ForecastDFDataset:
    """
    ForecastDFDataset 생성기 (버전 호환 + 런타임 오버라이드 지원)
    - context_len / prediction_len / known_real_cols 를 호출부에서 덮어쓸 수 있음
    """
    # 기본값: 글로벌 설정 사용
    context_len    = CONTEXT_LEN if context_len is None else int(context_len)
    prediction_len = PRED_LEN    if prediction_len is None else int(prediction_len)
    known_real_cols = KNOWN_REAL_COLS if known_real_cols is None else list(known_real_cols)

    sig = inspect.signature(ForecastDFDataset.__init__)
    params = set(sig.parameters.keys())
    kwargs = {}
    # 길이들
    if "context_length" in params:
        kwargs["context_length"] = context_len
    elif "context_len" in params:
        kwargs["context_len"] = context_len

    if "prediction_length" in params:
        kwargs["prediction_length"] = prediction_len
    elif "prediction_len" in params:
        kwargs["prediction_len"] = prediction_len

    if "stride" in params:
        kwargs["stride"] = DATA_STRIDE
    if "enable_padding" in params:
        kwargs["enable_padding"] = False

    # id / time / target
    for k in ["id_columns", "id_cols", "group_ids", "ids"]:
        if k in params:
            kwargs[k] = ID_COLS
            break

    for k in ["timestamp_column", "time_column", "time_col", "timestamp_col"]:
        if k in params:
            kwargs[k] = TIME_COL
            break

    for k in ["target_columns", "target_col", "target", "targets"]:
        if k in params:
            kwargs[k] = TARGET_COLS
            break

    # 동적 실수 피처 (채널 수를 바꿔야 할 때 여기로 제어)
    if "observable_columns" in params:
        kwargs["observable_columns"] = known_real_cols
    elif "control_columns" in params:
        kwargs["control_columns"] = known_real_cols
    elif "conditional_columns" in params:
        kwargs["conditional_columns"] = known_real_cols

    # 정적 범주 피처로 store_id 주입 (지원되는 파라미터 명에만 넣기)
    if "static_categorical_columns" in params:
        kwargs["static_categorical_columns"] = ["store_id"]
    elif "static_features" in params:  # 혹시 다른 이름을 쓰는 버전 대비
        kwargs["static_features"] = ["store_id"]

    if "num_workers" in params:
        kwargs["num_workers"] = 0

    if DEBUG_DATASET_SIG:
        print("[ForecastDFDataset accepted params]", sorted(params))
        print("[ForecastDFDataset kwargs]", kwargs)

    return ForecastDFDataset(df_split, **kwargs)

# ----------  # <CELL: collator>  (Insert)
import numpy as np
import pandas as pd
from transformers.data.data_collator import default_data_collator

# 배치에서 날짜·타임스탬프를 안전하게 처리(제거/정수화)
_DROP_KEYS_EXACT = {"date", "time", "start", "end", "target_start"}
_DROP_KEYS_SUBSTR = {"timestamp"}  # 키 이름에 'timestamp'가 들어가면 제거

def _to_int_ts(x):
    # pandas.Timestamp -> int64 (초 단위)
    return np.int64(x.value // 1_000_000_000)

def _sanitize_feature_dict(feat: dict):
    out = {}
    for k, v in feat.items():
        kl = k.lower()
        if kl in _DROP_KEYS_EXACT or any(sub in kl for sub in _DROP_KEYS_SUBSTR):
            # 모델 입력이 아닌 날짜 메타는 제거
            continue

        # 개별 Timestamp
        if isinstance(v, pd.Timestamp):
            out[k] = _to_int_ts(v)
            continue

        # 리스트에 Timestamp 포함
        if isinstance(v, list) and v and isinstance(v[0], pd.Timestamp):
            out[k] = np.array([_to_int_ts(t) for t in v], dtype=np.int64)
            continue

        # pandas Series -> numpy
        if isinstance(v, pd.Series):
            if np.issubdtype(v.dtype, np.datetime64):
                out[k] = v.view("i8") // 1_000_000_000
            else:
                out[k] = v.to_numpy()
            continue

        # numpy datetime64 배열
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.datetime64):
            out[k] = v.view("i8") // 1_000_000_000
            continue

        # 그 외(torch.Tensor/np.ndarray/수치형/리스트 등)는 그대로
        out[k] = v
    return out

def ts_data_collator(features):
    cleaned = [_sanitize_feature_dict(f) for f in features]
    return default_data_collator(cleaned)

# ----------  # <CELL: model>
import inspect
import torch
from torch import nn
from transformers import PatchTSTConfig, PatchTSTForPrediction

LEADERBOARD_OBJECTIVE = "SMAPE"

def _smape_torch(y_true, y_pred, eps=SMAPE_EPS):
    num = torch.abs(y_pred - y_true)
    den = (torch.abs(y_true) + torch.abs(y_pred)).clamp_min(eps)
    return 2.0 * (num / den)

def _mae_torch(y_true, y_pred):
    return torch.abs(y_pred - y_true)

def _choose_loss_weights(obj: str):
    # -> 기존 함수 확장: w_logmae, w_zero 추가
    obj = (obj or "").upper()
    if obj == "SMAPE":
        return dict(
            w_smape=SMAPE_WEIGHT,
            w_mae=MAE_WEIGHT,
            w_logmae=LOG_MAE_WEIGHT,
            w_zero=ZERO_OVERSHOOT_PENALTY
        )
    elif obj == "MAE":
        return dict(w_smape=0.0, w_mae=1.0, w_logmae=0.0, w_zero=0.0)
    else:
        return dict(w_smape=0.4, w_mae=0.4, w_logmae=0.2, w_zero=0.0)

class PatchTSTSalesOnly(nn.Module):
    """
    - 원 스케일 손실(SMAPE 중심) + 0-친화 보정:
      · y_true=0 타임스텝 다운웨이트
      · 전부0 윈도우 다운웨이트
      · log-MAE 보조항
      · zero-overshoot penalty(0에 양수 예측 억제)
      · (옵션) 특정 매장 가중(예: 미라시아/담하)
    """
    def __init__(self, base_model: PatchTSTForPrediction, target_ch: int = 0,
                 objective: str = LEADERBOARD_OBJECTIVE,
                 special_store_ids: set[int] | None = None):
        super().__init__()
        self.base = base_model
        self.target_ch = target_ch
        self._allowed = set(inspect.signature(self.base.forward).parameters.keys())
        self.loss_w = _choose_loss_weights(objective)

        # 가중 파라미터
        self.w_zero  = 1.05   # y_true==0 구간 샘플 가중 (과가중 완화)
        self.w_all0w = 1.00   # 윈도우 전체가 0일 때 가중

        self.special_store_ids = set(special_store_ids or [])

    def _filter_and_bridge(self, batch: dict):
        cleaned = {}
        for k, v in batch.items():
            if k in self._allowed:
                cleaned[k] = v
        if "labels" in self._allowed and "labels" not in cleaned and "future_values" in batch and "future_values" not in self._allowed:
            cleaned["labels"] = batch["future_values"]
        if "observed_mask" in self._allowed and "observed_mask" not in cleaned:
            po = batch.get("past_observed_mask", None)
            fo = batch.get("future_observed_mask", None)
            if po is not None and fo is not None:
                try: cleaned["observed_mask"] = torch.cat([po, fo], dim=-1)
                except Exception: cleaned["observed_mask"] = po
            elif po is not None:
                cleaned["observed_mask"] = po
        return cleaned

    def forward(self, **batch):
        cleaned = self._filter_and_bridge(batch)
        out = self.base(**cleaned)

        pred = getattr(out, "prediction", None)
        if pred is not None and pred.dim() == 3:
            if pred.shape[1] == self.base.config.prediction_length:
                # (B, L, C) -> select channel
                ch = min(self.target_ch, pred.shape[2]-1)
                pred = pred[:, :, ch]
            else:
                # (B, C, L)
                ch = min(self.target_ch, pred.shape[1]-1)
                pred = pred[:, ch, :]

        labels = cleaned.get("labels", None)
        if pred is not None and labels is not None:
            eps   = 1e-6
            yhat  = torch.expm1(pred).clamp_min(0)
            ytrue = torch.expm1(labels).clamp_min(0)

            w_pos  = (ytrue > 0).float()
            w_zero = 1.0 - w_pos
            w_t = self.w_zero * w_zero + 1.0 * w_pos

            all0 = (ytrue.sum(dim=-1, keepdim=True) == 0).float()
            w_w  = self.w_all0w * all0 + 1.0 * (1.0 - all0)

            w_s = torch.ones_like(ytrue)
            scf = cleaned.get("static_categorical_features", None)
            if scf is None:
                scf = batch.get("static_categorical_features", None)
            if scf is None:
                scf = batch.get("static_features", None)  # 일부 구현체 호환

            if scf is not None:
                sid = scf.squeeze(-1) if scf.dim()==2 else scf  # (B,)
                if self.special_store_ids:
                    m = torch.zeros_like(sid, dtype=torch.float32)
                    for s in self.special_store_ids:
                        m = m + (sid == s).float()
                    m = (m > 0).float().unsqueeze(-1).expand_as(ytrue)
                    w_s = torch.where(m>0, torch.tensor(2.0, device=ytrue.device), torch.tensor(1.0, device=ytrue.device))

            W = w_t * w_w * w_s

            smape    = _smape_torch(ytrue, yhat)
            mae      = _mae_torch(ytrue, yhat)
            log_mae  = torch.abs(torch.log1p(ytrue+eps) - torch.log1p(yhat+eps))
            overshot = torch.relu(yhat) * (ytrue == 0).float()

            w = self.loss_w
            loss = 0.0
            if w.get("w_smape", 0) > 0:
                loss += w["w_smape"] * torch.sum(W * smape)   / (W.sum() + eps)
            if w.get("w_mae", 0) > 0:
                loss += w["w_mae"]   * torch.sum(W * mae)     / (W.sum() + eps)
            if w.get("w_logmae", 0) > 0:
                loss += w["w_logmae"] * torch.sum(W * log_mae) / (W.sum() + eps)
            if w.get("w_zero", 0) > 0:  # ← overshoot penalty 계수 (이름만 'zero')
                loss += w["w_zero"] * torch.sum(W * overshot) / (W.sum() + eps)

            out.loss = loss

        if pred is not None:
            out.prediction = pred
        return out

def make_model(context_len: int):
    # context_len을 인자로 받아서 구성
    config = PatchTSTConfig(
        num_input_channels=1 + len(KNOWN_REAL_COLS),
        context_length=int(context_len),
        prediction_length=PRED_LEN,
        patch_length=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
        d_model=320,
        num_attention_heads=8,
        num_hidden_layers=6,
        ffn_dim=640,
        dropout=0.15,
        head_dropout=0.15,
        scaling="std",
        loss="mse",
    )
    base = PatchTSTForPrediction(config)
    return PatchTSTSalesOnly(base, target_ch=0, objective=SPLIT_OBJECTIVE)


# ----------  # <CELL: metrics>  (NEW)

def _flatten_any(x):
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            out.extend(_flatten_any(e))
        return out
    return [np.asarray(x)]

def _extract_matrix_from_any(raw, pred_len: int, target_ch: int = 0):
    arrs = _flatten_any(raw)
    nlc, ncl = [], []
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 3:
            if a.shape[1] == pred_len:   # (N, L, C)
                nlc.append(a)
            elif a.shape[2] == pred_len: # (N, C, L)
                ncl.append(a)
    if nlc:
        a = nlc[0]; ch = target_ch if a.shape[2] > target_ch else 0
        return a[:, :, ch]
    if ncl:
        a = ncl[0]; ch = target_ch if a.shape[1] > target_ch else 0
        return a[:, ch, :]
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == pred_len:
            return a
    raise ValueError(f"cannot shape to (N,{pred_len})")

def _smape_np(y, yhat, eps=1e-6):
    num = np.abs(yhat - y)
    den = (np.abs(y) + np.abs(yhat) + eps)
    return 2.0 * (num / den)

def compute_metrics(eval_pred):
    # 로그 → 원 스케일
    Yhat_log = _extract_matrix_from_any(eval_pred.predictions, pred_len=PRED_LEN, target_ch=0)
    Ylbl_log = _extract_matrix_from_any(eval_pred.label_ids,   pred_len=PRED_LEN, target_ch=0)

    yhat = np.clip(np.expm1(Yhat_log), 0, None)
    ytrue = np.clip(np.expm1(Ylbl_log), 0, None)

    mae   = float(np.mean(np.abs(yhat - ytrue)))
    rmse  = float(np.sqrt(np.mean((yhat - ytrue) ** 2)))
    smape = float(np.mean(_smape_np(ytrue, yhat)))

    return {"mae": mae, "rmse": rmse, "smape": smape}

# ----------  # <CELL: training args>
import inspect

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
OPTIM_NAME = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

# 1) 현재 TrainingArguments가 어떤 파라미터를 받는지 확인
sig = inspect.signature(TrainingArguments.__init__)
PARAMS = set(sig.parameters.keys())

# 2) 공통(모든 버전에서 문제없는) 기본 kwargs
kw = dict(
    output_dir=SAVE_DIR,
    overwrite_output_dir=True,

    num_train_epochs=30,
    do_eval=True,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    label_names=["future_values"],
    remove_unused_columns=False,

    dataloader_pin_memory=True,
    report_to="none",
)

# 3) 버전별 옵션을 "있을 때만" 추가
if "evaluation_strategy" in PARAMS:
    kw["evaluation_strategy"] = "epoch"
elif "eval_strategy" in PARAMS:
    kw["eval_strategy"] = "epoch"

if "dataloader_num_workers" in PARAMS:
    kw["dataloader_num_workers"] = (4 if os.name != "nt" else 0)
    # dataloader_num_workers : 리눅스/WSL : 4 / 8, Windows : 0 유지

if "dataloader_persistent_workers" in PARAMS:
    kw["dataloader_persistent_workers"] = (os.name != "nt")

if "tf32" in PARAMS:
    kw["tf32"] = True

if "bf16" in PARAMS:
    kw["bf16"] = USE_BF16

if "fp16" in PARAMS:
    kw["fp16"] = not USE_BF16

if "optim" in PARAMS:
    kw["optim"] = OPTIM_NAME

if "eval_accumulation_steps" in PARAMS:
    kw["eval_accumulation_steps"] = 32


if "learning_rate" in PARAMS:
    kw["learning_rate"] = 4e-4
    # 3e-4~1e-3 권역; 5e-4 무난 / 지금 세팅(배치 64, AdamW-fused, warmup_ratio=0.05, stride=1) 기준 learning_rate=4e-4가 1차 권장값입니다.
if "weight_decay" in PARAMS:
    kw["weight_decay"]  = 0.01
if "warmup_ratio" in PARAMS:
    kw["warmup_ratio"]  = 0.05

# 선택 지표: SMAPE로 고정
select_metric = "eval_smape"
kw["load_best_model_at_end"] = True
kw["metric_for_best_model"]  = select_metric
kw["greater_is_better"]      = False

# 4) 최종 생성
training_args = TrainingArguments(**kw)
print("[TrainingArguments OK]")
print("Accepted keys:", sorted(kw.keys()))

EARLY_STOP_PATIENCE = 6

# ----------  # <CELL: callbacks (anchor eval)>
class RotateEvalAnchors(TrainerCallback):
    """매 epoch 평가 셋을 7일 간격 서브샘플로 교체(속도↑, 성능 영향 없음)."""
    def __init__(self, trainer_ref, full_eval_ds, step: int = ANCHOR_STEP):
        self.trainer = trainer_ref
        self.full_eval_ds = full_eval_ds
        self.step = step
    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch) if state.epoch is not None else 0
        off = ep % self.step
        idx = list(range(off, len(self.full_eval_ds), self.step))
        if not idx:  # ★ 최소 1개 보장
            idx = [0]
        self.trainer.eval_dataset = Subset(self.full_eval_ds, idx)
        print(f"[RotateEvalAnchors] epoch={ep} offset={off} eval_size={len(idx)}")

# ----------  # <CELL: build dataframes>
raw = load_train_df()
raw = enforce_regular_daily(raw)
raw = add_time_features(raw)      # 캘린더/휴일 피처
raw = add_rolling_channels(raw)   # 🔥 롤링 채널 추가

le  = fit_or_load_label_encoder(raw["store_menu"])
df  = finalize_columns(raw, le)

N_WEEKS = int(df["week_idx"].max()) + 1
print(f"Rows={len(df)}, Items={df['store_menu_id'].nunique()}, Weeks={N_WEEKS}")

# REPLACE ↓ (상한 계산을 원본 기반으로 교체)
ITEM_CAP = build_item_caps_from_original()

# ADD ↓ 특정 매장 id 자동 수집(옵션)
def collect_special_store_ids(df_: pd.DataFrame) -> set[int]:
    s = set()
    for nm, sid in df_[["store_name","store_id"]].drop_duplicates().itertuples(index=False, name=None):
        if ("미라시아" in nm) or ("Miracia" in nm) or ("담하" in nm) or ("Damha" in nm):
            s.add(int(sid))
    return s

SPECIAL_STORE_IDS = collect_special_store_ids(df)
print("Special store ids:", SPECIAL_STORE_IDS)

# ----------  # <CELL: cv split & run>
# 연속 주(week_idx) 기준 K-Fold (불균등 분할도 커버)
def contiguous_week_folds(weeks_sorted, k):
    # np.array_split으로 연속 블록 K개로 나눔
    return [list(chunk) for chunk in np.array_split(weeks_sorted, k)]

def make_masks_by_weeks(valid_weeks, all_weeks, purge_gap=1):
    valid_weeks = set(valid_weeks)
    if len(valid_weeks) == 0:
        raise ValueError("valid_weeks is empty.")
    min_w, max_w = min(valid_weeks), max(valid_weeks)
    # purge 범위: [min_w - gap, max_w + gap]
    purge_range = set([w for w in all_weeks if (min_w - purge_gap) <= w <= (max_w + purge_gap)])
    w_arr = df["week_idx"].values
    valid_mask = np.isin(w_arr, list(valid_weeks))
    purge_mask = np.isin(w_arr, list(purge_range))
    train_mask = (~valid_mask) & (~purge_mask)
    return train_mask, valid_mask

all_weeks_sorted = sorted(df["week_idx"].unique().tolist())

# 컨텍스트별 CV 결과/가중치 저장
FOLD_WEIGHTS_CTX = {}   # {ctx: [w_fold_0, ...]}
CV_SUMMARY_CTX   = {}   # {ctx: {"loss":..,"smape":..,"mae":..,"rmse":..,"fold_weights":[..]}}

for CTX in CONTEXT_SET:
    fold_weeks = contiguous_week_folds(all_weeks_sorted, K_FOLDS)
    cv_metrics = []
    print(f"\n========== [CV @ context={CTX}] ==========")

    for fold, v_weeks in enumerate(fold_weeks):
        tr_m, va_m = make_masks_by_weeks(v_weeks, all_weeks_sorted, purge_gap=PURGE_GAP_WEEKS)
        train_df = df.loc[tr_m].copy()
        valid_df = df.loc[va_m].copy()

        train_ds = build_dataset(train_df, context_len=CTX, prediction_len=PRED_LEN)
        valid_ds = build_dataset(valid_df, context_len=CTX, prediction_len=PRED_LEN)

        model = make_model(CTX)
        model.special_store_ids = SPECIAL_STORE_IDS  # 특수 매장 가중

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=ts_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
        )
        trainer.add_callback(RotateEvalAnchors(trainer, valid_ds, step=ANCHOR_STEP))

        print(f"\n[CV] ctx={CTX} fold={fold} train_rows={len(train_df)} valid_rows={len(valid_df)} weeks={min(v_weeks)}..{max(v_weeks)}")
        trainer.train()
        trainer.eval_dataset = valid_ds
        m = trainer.evaluate()

        # 컨텍스트별 체크포인트 폴더
        fold_dir = os.path.join(ctx_save_dir(CTX), f"fold_{fold}")
        trainer.save_model(fold_dir)

        m["fold"] = fold
        cv_metrics.append(m)
        print(f"[CV] ctx={CTX} fold={fold} metrics={m}")

    # 컨텍스트별 CV 요약/가중치
    cv_eval_loss = float(np.mean([m["eval_loss"] for m in cv_metrics]))
    cv_smape = float(np.mean([m.get("eval_smape", np.nan) for m in cv_metrics]))
    cv_mae   = float(np.mean([m.get("eval_mae",   np.nan) for m in cv_metrics]))
    cv_rmse  = float(np.mean([m.get("eval_rmse",  np.nan) for m in cv_metrics]))
    print(f"[CV ctx={CTX}] avg → loss={cv_eval_loss:.6f}, smape={cv_smape:.6f}, mae={cv_mae:.3f}, rmse={cv_rmse:.3f}")

    fold_smapes = [m.get("eval_smape", np.inf) for m in cv_metrics]
    if all(np.isfinite(s) for s in fold_smapes) and len(fold_smapes) > 0:
        w = 1.0 / (np.asarray(fold_smapes) + 1e-6)
        FOLD_WEIGHTS_CTX[CTX] = (w / w.sum()).astype(float).tolist()
    else:
        FOLD_WEIGHTS_CTX[CTX] = [1.0 / max(1, len(cv_metrics))] * max(1, len(cv_metrics))

    CV_SUMMARY_CTX[CTX] = dict(loss=cv_eval_loss, smape=cv_smape, mae=cv_mae, rmse=cv_rmse,
                               fold_weights=FOLD_WEIGHTS_CTX[CTX])

print("\n[FOLD ENSEMBLE] per-context weights:")
for k, v in FOLD_WEIGHTS_CTX.items():
    print(f"  ctx={k} → {v}")

# (NEW) Save fold weights for reuse at inference time
import json
w_path = os.path.join(SAVE_DIR, "fold_weights_ctx.json")
with open(w_path, "w", encoding="utf-8") as f:
    json.dump(FOLD_WEIGHTS_CTX, f, ensure_ascii=False, indent=2)
print(f"[SAVE] fold weights saved to {w_path}")

# ----------  # <CELL: final fit (all data)>
FINAL_METRICS_CTX = {}

for CTX in CONTEXT_SET:
    print(f"\n========== [FINAL TRAIN @ context={CTX}] ==========")
    train_all = build_dataset(df, context_len=CTX, prediction_len=PRED_LEN)
    valid_all = build_dataset(df, context_len=CTX, prediction_len=PRED_LEN)

    final_model = make_model(CTX)
    final_model.special_store_ids = SPECIAL_STORE_IDS

    final_trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=train_all,
        eval_dataset=valid_all,
        data_collator=ts_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    final_trainer.add_callback(RotateEvalAnchors(final_trainer, valid_all, step=ANCHOR_STEP))

    final_trainer.train()
    final_trainer.eval_dataset = valid_all
    final_metrics = final_trainer.evaluate()
    FINAL_METRICS_CTX[CTX] = final_metrics
    print(f"[FINAL ctx={CTX}] eval:", final_metrics)

    out_dir = os.path.join(ctx_save_dir(CTX), "best")
    os.makedirs(out_dir, exist_ok=True)
    final_trainer.save_model(out_dir)
    print("Saved:", out_dir)

print("\n[FINAL METRICS by context]")
for k, v in FINAL_METRICS_CTX.items():
    print(f"  ctx={k} → {v}")
print("LabelEncoder:", LE_PATH)

# ----------  # <CELL: load fold models for inference ensemble>  (REPLACE WHOLE CELL)

from copy import deepcopy

INFER_TRAINERS_CTX = {}  # {ctx: [Trainer, ...]}

# (NEW) Try to load saved fold weights (resume-friendly)
try:
    import json
    w_path = os.path.join(SAVE_DIR, "fold_weights_ctx.json")
    if os.path.exists(w_path):
        with open(w_path, "r", encoding="utf-8") as f:
            _saved = json.load(f)
        # keys가 문자열일 수 있으니 int로 강제 변환
        FOLD_WEIGHTS_CTX = {int(k): v for k, v in _saved.items()}
        print(f"[LOAD] fold weights loaded from {w_path}")
except Exception as e:
    print(f"[LOAD] fold weights JSON read failed: {e}")

if FOLD_ENSEMBLE:
    for CTX in CONTEXT_SET:
        trainers = []
        ctx_dir = ctx_save_dir(CTX)
        for fold in range(K_FOLDS):
            fold_dir = os.path.join(ctx_dir, f"fold_{fold}")
            # 둘 중 존재하는 걸 로드 (.safetensors 우선)
            bin_path = None
            for fn in ["model.safetensors", "pytorch_model.bin"]:
                p = os.path.join(fold_dir, fn)
                if os.path.exists(p):
                    bin_path = p
                    break

            if bin_path is not None:
                # ★ no-grad 컨텍스트: 모델 생성/로드/모드전환까지 불필요한 그래프 방지
                with torch.inference_mode():
                    m = make_model(CTX)
                    m.special_store_ids = SPECIAL_STORE_IDS
                    if bin_path.endswith(".safetensors"):
                        from safetensors.torch import load_file as safe_load

                        sd = safe_load(bin_path)
                    else:
                        sd = torch.load(bin_path, map_location="cpu")

                    missing, unexpected = m.load_state_dict(sd, strict=False)
                    m.eval()

                if missing:    print(f"[ctx {CTX} fold {fold}] missing keys:", len(missing))
                if unexpected: print(f"[ctx {CTX} fold {fold}] unexpected keys:", len(unexpected))

                # 🔑 추론 전용 TrainingArguments 복사/무력화
                infer_args = deepcopy(training_args)
                if hasattr(infer_args, "evaluation_strategy"): infer_args.evaluation_strategy = "no"
                if hasattr(infer_args, "eval_strategy"):       infer_args.eval_strategy       = "no"
                infer_args.do_eval                = False
                infer_args.load_best_model_at_end = False
                if hasattr(infer_args, "save_strategy"):       infer_args.save_strategy       = "no"
                infer_args.report_to              = "none"
                # 워커/퍼시스턴트 비활성화(추론 안정)
                if hasattr(infer_args, "dataloader_num_workers"):        infer_args.dataloader_num_workers = 0
                if hasattr(infer_args, "dataloader_persistent_workers"): infer_args.dataloader_persistent_workers = False

                # (가드) collator 미정의 상태로 재개했을 때 대비
                if 'ts_data_collator' not in globals():
                    from transformers.data.data_collator import default_data_collator as ts_data_collator

                t = Trainer(
                    model=m,
                    args=infer_args,
                    data_collator=ts_data_collator,
                )
                trainers.append(t)
            else:
                print(f"[FOLD ENSEMBLE] checkpoint not found in {fold_dir}")

        INFER_TRAINERS_CTX[CTX] = trainers
        print(f"[FOLD ENSEMBLE] loaded ctx={CTX}: {len(trainers)} fold models.")
else:
    # 폴드 앙상블 OFF: 컨텍스트별 'best' 하나씩 로드
    for CTX in CONTEXT_SET:
        best_dir = os.path.join(ctx_save_dir(CTX), "best")
        bin_path = None
        for fn in ["model.safetensors", "pytorch_model.bin"]:
            p = os.path.join(best_dir, fn)
            if os.path.exists(p):
                bin_path = p
                break
        trainers = []
        if bin_path is not None:
            with torch.set_grad_enabled(False):
                m = make_model(CTX)
                m.special_store_ids = SPECIAL_STORE_IDS
                if bin_path.endswith(".safetensors"):
                    from safetensors.torch import load_file as safe_load

                    sd = safe_load(bin_path)
                else:
                    sd = torch.load(bin_path, map_location="cpu")
                m.load_state_dict(sd, strict=False)
                m.eval()

            infer_args = deepcopy(training_args)
            if hasattr(infer_args, "evaluation_strategy"): infer_args.evaluation_strategy = "no"
            if hasattr(infer_args, "eval_strategy"):       infer_args.eval_strategy       = "no"
            infer_args.do_eval                = False
            infer_args.load_best_model_at_end = False
            if hasattr(infer_args, "save_strategy"):       infer_args.save_strategy       = "no"
            infer_args.report_to              = "none"
            if hasattr(infer_args, "dataloader_num_workers"):        infer_args.dataloader_num_workers = 0
            if hasattr(infer_args, "dataloader_persistent_workers"): infer_args.dataloader_persistent_workers = False

            if 'ts_data_collator' not in globals():
                from transformers.data.data_collator import default_data_collator as ts_data_collator

            t = Trainer(model=m, args=infer_args, data_collator=ts_data_collator)
            trainers.append(t)
        INFER_TRAINERS_CTX[CTX] = trainers
        print(f"[SINGLE] loaded ctx={CTX}: {len(trainers)} model(s).")

# ----------  # <CELL: inference> (Replace: helper 포함, sample_submission 저장)

import os, re, gc, glob
import numpy as np
import pandas as pd
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# (NEW) 최근 K일이 전부 0이면 미래 7일 0으로 고정하는 가드
ZERO_RUN_GUARD_DAYS = globals().get("ZERO_RUN_GUARD_DAYS", 14)

# (존재하지 않으면 기본값 세팅 - 네 코드랑 변수명 호환)
USE_INT_ROUND    = globals().get("USE_INT_ROUND", False)   # 정수 제출 아님: False 권장
CUT_THRESHOLD    = globals().get("CUT_THRESHOLD", None)    # 예: 0.9 등, None이면 미사용
ENSEMBLE_NAIVE_W = globals().get("ENSEMBLE_NAIVE_W", 0.20) # 보수적 앙상블 가중(0~1)
CAP_MULT         = globals().get("CAP_MULT", 1.0)          # 상한 여유 배수
ITEM_CAP         = globals().get("ITEM_CAP", {}) or {}

# (필요 시) 간단한 leftpad 구현 — 네가 이미 정의해둔 함수가 있으면 그걸 사용
if "leftpad_to_context" not in globals():
    def leftpad_to_context(g: pd.DataFrame, context_len: int, store_menu: str) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        n = len(g)
        if n >= context_len:
            return g
        need = context_len - n
        pad_end = g["date"].min() - pd.Timedelta(days=1)
        pad_dates = pd.date_range(end=pad_end, periods=need, freq="D")
        store, menu = store_menu.split("_", 1)
        pad = pd.DataFrame({
            "date": pad_dates,
            "영업일자": pad_dates,
            "store_name": store,
            "menu_name": menu,
            "store_menu": store_menu,
            "영업장명_메뉴명": store_menu,
            "매출수량": 0.0,
            "sales": 0.0,
        })
        pad = add_time_features(pad)
        return pd.concat([pad, g], ignore_index=True)

# --- helper: predictions -> (N, pred_len)로 정규화 ---
def _extract_pred_matrix(pred_output, pred_len: int, target_ch: int = 0):
    raw = pred_output.predictions
    def _flatten(x):
        if isinstance(x, (list, tuple)):
            out = []
            for e in x: out.extend(_flatten(e))
            return out
        return [np.asarray(x)]

    arrs = _flatten(raw)
    nlc_candidates, ncl_candidates = [], []
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 3:
            if a.shape[1] == pred_len:   # (N, L, C)
                nlc_candidates.append(a)
            elif a.shape[2] == pred_len: # (N, C, L)
                ncl_candidates.append(a)
    if nlc_candidates:
        a = nlc_candidates[0]
        ch = target_ch if a.shape[2] > target_ch else 0
        return a[:, :, ch]
    if ncl_candidates:
        a = ncl_candidates[0]
        ch = target_ch if a.shape[1] > target_ch else 0
        return a[:, ch, :]
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == pred_len:
            return a
    shapes = [np.asarray(a).shape for a in arrs]
    raise ValueError(f"[extract] Cannot find (N,{pred_len}) from predictions; seen shapes={shapes}")

def _find_sample_submission():
    for p in ["./dataset/sample_submission.csv", "./sample_submission.csv", "/mnt/data/sample_submission.csv"]:
        if os.path.exists(p): return p
    raise FileNotFoundError("sample_submission.csv 경로를 찾지 못했습니다.")

def _make_future_rows(store_menu, last_date, horizon=PRED_LEN):
    # 미래 1~horizon일 생성(모델 입력용 공변량만 필요)
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    store, menu = str(store_menu).split("_", 1)
    fut = pd.DataFrame({
        "영업일자": future_dates,
        "영업장명_메뉴명": store_menu,
        "매출수량": np.nan,
        "store_name": store,
        "menu_name": menu,
        "store_menu": store_menu,
        "date": future_dates,
    })
    fut = add_time_features(fut)
    return fut

def _predict_last_window_for_file(file_path):
    """
    TEST_xx.csv 하나에 대해:
      - 각 '영업장명_메뉴명'별 마지막 윈도우(CTX별) 구성
      - 컨텍스트×폴드 앙상블 + (가능하면) 컨텍스트 적응형 가중
      - baseline 포맷(영업일자='TEST_xx+{d}일')으로 레코드 반환
    """
    import re
    test_prefix = re.search(r"(TEST_\d+)", os.path.basename(file_path)).group(1)

    df_t = pd.read_csv(file_path)
    df_t["date"] = pd.to_datetime(df_t["영업일자"])
    s = df_t["영업장명_메뉴명"].astype(str).str.split("_", n=1, expand=True)
    df_t["store_name"] = s[0]; df_t["menu_name"] = s[1]
    df_t["store_menu"] = df_t["store_name"] + "_" + df_t["menu_name"]
    df_t["매출수량"] = pd.to_numeric(df_t["매출수량"], errors="coerce").fillna(0.0)
    df_t.loc[df_t["매출수량"] < 0, "매출수량"] = 0.0
    df_t["sales"] = df_t["매출수량"].astype(float)

    df_t = enforce_regular_daily(df_t)
    df_t = add_time_features(df_t)

    # ---- (안전) 컨텍스트 가중 헬퍼: 외부 정의 없으면 내부 fallback 사용
    def _weekly_strength_and_nz_local(g_: pd.DataFrame) -> tuple[float, float]:
        v = pd.to_numeric(g_["매출수량"], errors="coerce").fillna(0.0).to_numpy()
        v28 = v[-28:] if v.size >= 28 else v
        nz28 = float((v28 > 0).mean()) if v28.size else 0.0
        try:
            dow = g_["weekday"].to_numpy()[-len(v28):]
            if v28.size >= 14 and np.var(v28) > 0:
                means = [v28[dow == d].mean() for d in range(7) if (dow == d).any()]
                weekly_strength = float(np.var(means) / (np.var(v28) + 1e-9)) if len(means) >= 2 else 0.0
            else:
                weekly_strength = 0.0
        except Exception:
            weekly_strength = 0.0
        return float(np.clip(weekly_strength, 0.0, 1.0)), nz28

    def _context_mix_weights_fallback(g_: pd.DataFrame, contexts: list[int]) -> dict[int, float]:
        # 외부 _context_mix_weights가 있으면 그걸 사용
        if "_context_mix_weights" in globals() and callable(globals()["_context_mix_weights"]):
            try:
                return globals()["_context_mix_weights"](g_, contexts)
            except Exception:
                pass
        # 내부 간단 적응형: base + (요일성, 희소도) 보정
        base = {28: 0.35, 56: 0.40, 84: 0.25}
        w = np.array([base.get(c, 0.0) for c in contexts], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(contexts), dtype=float) / len(contexts)
        weekly_strength, nz28 = _weekly_strength_and_nz_local(g_)
        adj = np.zeros_like(w)
        for i, c in enumerate(contexts):
            adj[i] += (0.15 * weekly_strength) * (1 if c >= 56 else -1)
            adj[i] += (0.10 * (1.0 - nz28))   * (1 if c == 28 else (-1 if c >= 84 else 0))
        w = np.clip(w + adj, 1e-6, None); w = (w / w.sum()).astype(float)
        return {c: float(w[i]) for i, c in enumerate(contexts)}

    records = []

    for store_menu, g in df_t.groupby("영업장명_메뉴명"):
        g_raw = g.sort_values("date").copy()

        # 아이템별 컨텍스트 가중(적응형)
        ctx_weights = _context_mix_weights_fallback(g_raw, CONTEXT_SET)

        yhat_ctx = {}  # {ctx: (7,) 예측}
        for CTX in CONTEXT_SET:
            # 입력 구성
            g_ctx = leftpad_to_context(g_raw, CTX, store_menu)
            last_date = g_ctx["date"].max()
            fut = _make_future_rows(store_menu, last_date, horizon=PRED_LEN)
            combo = pd.concat([g_ctx, fut], ignore_index=True)
            combo["sales"] = pd.to_numeric(combo["매출수량"], errors="coerce").fillna(0.0)
            combo = add_rolling_channels(combo)  # 미래구간도 과거통계 carry-forward
            # 캐리-포워드: 미래 구간(원본 매출수량 NaN)은 마지막 관측값으로 고정
            fmask = combo["매출수량"].isna()
            if fmask.any():
                chs = ["roll7_mean", "roll28_mean", "roll7_med", "nzrate28"]
                if (~fmask).any():
                    last_obs_idx = combo.index[~fmask][-1]
                    for c in chs:
                        combo.loc[fmask, c] = float(combo.loc[last_obs_idx, c])
                else:
                    # 전부 미래(혹은 패딩만)인 극단 케이스: 0으로
                    for c in chs:
                        combo.loc[fmask, c] = 0.0

            combo_tail = combo.iloc[-(CTX + PRED_LEN):].copy()

            # 라벨인코더(테스트 신규 ID 포함)
            base_series = raw["store_menu"] if "raw" in globals() else combo_tail["store_menu"]
            le2 = fit_or_load_label_encoder(pd.concat([base_series, combo_tail["store_menu"]]))
            combo_fin = finalize_columns(combo_tail, le2)

            ds = build_dataset(combo_fin, context_len=CTX, prediction_len=PRED_LEN)

            # 이 컨텍스트의 fold 모델들
            predictors = INFER_TRAINERS_CTX.get(CTX, [])
            if len(predictors) == 0:
                print(f"[WARN] no predictors for ctx={CTX}; skip this ctx")
                continue

            fold_w = np.asarray(FOLD_WEIGHTS_CTX.get(CTX, []), dtype=float)
            if fold_w.size != len(predictors) or not np.isfinite(fold_w).all():
                fold_w = np.ones(len(predictors), dtype=float) / len(predictors)

            yhat_list = []
            for t in predictors:
                # predict 순간만 워커/배치/퍼시스턴트 워커 설정 임시 조정
                old_workers = getattr(t.args, "dataloader_num_workers", None)
                old_eval_bs = getattr(t.args, "per_device_eval_batch_size", None)
                old_persist = getattr(t.args, "dataloader_persistent_workers", None)

                t.args.dataloader_num_workers = 0
                t.args.per_device_eval_batch_size = min(16, (old_eval_bs or 16))
                if old_persist is not None:
                    t.args.dataloader_persistent_workers = False

                try:
                    with torch.inference_mode():
                        preds_out = t.predict(ds)
                finally:
                    if old_workers is not None:
                        t.args.dataloader_num_workers = old_workers
                    if old_eval_bs is not None:
                        t.args.per_device_eval_batch_size = old_eval_bs
                    if old_persist is not None:
                        t.args.dataloader_persistent_workers = old_persist

                # 로그->원 스케일 & 마지막 샘플 7일만
                Y = _extract_pred_matrix(preds_out, PRED_LEN, target_ch=0)
                yhat_i = np.clip(np.expm1(Y[-1]), 0, None)
                yhat_list.append(yhat_i)

            S = np.stack(yhat_list, axis=0)           # (n_folds, 7)
            yhat_fold_avg = np.average(S, axis=0, weights=fold_w)  # (7,)
            yhat_ctx[CTX] = yhat_fold_avg

        if not yhat_ctx:
            # 모든 컨텍스트 실패 시 0
            yhat = np.zeros(PRED_LEN, dtype=float)
        else:
            ctxs = sorted(yhat_ctx.keys())
            W = np.array([ctx_weights.get(c, 0.0) for c in ctxs], dtype=float)
            if not np.isfinite(W).all() or W.sum() <= 0:
                W = np.ones(len(ctxs), dtype=float) / len(ctxs)
            M = np.stack([yhat_ctx[c] for c in ctxs], axis=0)  # (n_ctx, 7)
            yhat = np.average(M, axis=0, weights=W)

        # 나이브 블렌딩
        if ENSEMBLE_NAIVE_W > 0:
            yhat = _blend_with_naive(yhat, g_raw)

        # 제로-런 가드 & 캡
        yhat = _zero_run_guard(g_raw, yhat)
        cap = ITEM_CAP.get(store_menu, None)
        if cap is not None and np.isfinite(cap):
            yhat = np.minimum(yhat, float(cap) * float(CAP_MULT))

        # 출력 적재
        pred_dates = [f"{test_prefix}+{i + 1}일" for i in range(PRED_LEN)]
        for d_str, val in zip(pred_dates, yhat):
            records.append({
                "영업일자": d_str,
                "영업장명_메뉴명": store_menu,
                "매출수량": float(val),
            })

    return pd.DataFrame(records)

# --- 실행: TEST_*별 예측 → sample_submission으로 피벗 & 저장 ---
test_files = find_test_files()
if not test_files:
    print("No TEST_*.csv detected; skipping inference.")
else:
    print("Found test files:", len(test_files))
    ss_path = _find_sample_submission()
    submit_df = pd.read_csv(ss_path)  # 최종 제출 DF (여기에 채워넣음)

    # 숫자 컬럼 float로 열어둠 (마지막에 실수/정수 토글)
    for c in submit_df.columns[1:]:
        submit_df[c] = 0.0

    for p in sorted(test_files):
        df_pred_one = _predict_last_window_for_file(p)

        # 이 파일에 해당하는 행만 선택 (예: 'TEST_03+1일' 등)
        pred_index = df_pred_one["영업일자"].unique().tolist()
        mask_rows = submit_df["영업일자"].isin(pred_index)

        # 파일 단위 피벗: (행=영업일자, 열=아이템, 값=수량)
        pivot = df_pred_one.pivot(index="영업일자",
                                  columns="영업장명_메뉴명",
                                  values="매출수량")

        # 공통 열만 주입 (간혹 열 불일치 대비)
        common_cols = submit_df.columns[1:].intersection(pivot.columns)
        sub_view = pivot.reindex(index=submit_df.loc[mask_rows, "영업일자"],
                                 columns=common_cols).fillna(0.0)

        # 값 대입(넘파이로 복사 → 메모리 절약)
        submit_df.loc[mask_rows, common_cols] = sub_view.to_numpy()

        # 메모리 정리
        del df_pred_one, pivot, sub_view
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Filled rows for {os.path.basename(p)} → {mask_rows.sum()} rows, cols={len(common_cols)}")

    # 사전 점검: 0 비율
    zero_ratio_before = (submit_df.iloc[:, 1:] == 0).mean().mean()
    print(f"[Sanity] zero ratio BEFORE postprocess: {zero_ratio_before:.3f}")

    # 후처리: 컷오프/정수화 토글
    vals = submit_df.iloc[:, 1:].to_numpy(dtype=float, copy=False)
    if CUT_THRESHOLD is not None:
        vals = np.where(vals < float(CUT_THRESHOLD), 0.0, vals)
    if USE_INT_ROUND:
        vals = np.rint(np.clip(vals, 0, None)).astype(int, copy=False)
    else:
        vals = np.clip(vals, 0, None)  # 실수 유지(정수 제출 아님)
    submit_df.iloc[:, 1:] = vals

    zero_ratio_after = (submit_df.iloc[:, 1:] == 0).mean().mean()
    print(f"[Sanity] zero ratio AFTER postprocess: {zero_ratio_after:.3f}")

    out_dir = os.path.join(SAVE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission_patchtst.csv")
    submit_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("제출용 CSV 저장 완료 →", out_path)