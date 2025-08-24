# ----------  # <CELL: imports & device>
import os, glob, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import holidays

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
PRED_LEN    = 7
PATCH_LEN   = 7
PATCH_STRIDE= 7         # 7 / 1
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
SUBMISSION_ROUND_INT = True    # 규정이 정수 필수면 True 유지
SMALL_VALUE_CUTOFF  = 0.0      # 0.9 등으로 두면 그 미만은 0 강제
FOLD_ENSEMBLE = True           # 폴드 앙상블 추론 활성화

# Loss 가중치(원-스케일 sMAPE 중심 + log-MAE 보강 + 0-overshoot 패널티)
SPLIT_OBJECTIVE = "SMAPE"   # 기존 LEADERBOARD_OBJECTIVE와 의미 동일
SMAPE_WEIGHT    = 0.7
MAE_WEIGHT      = 0.0       # zero-heavy 데이터면 원-MAE 비중은 낮추는 게 sMAPE에 유리
LOG_MAE_WEIGHT  = 0.3       # log-space 안정화(저수량/제로 근처 진동 억제)
SMAPE_EPS       = 1e-6      # sMAPE 분모 안정화용(원한다면 1e-5~1e-4로 상향 테스트)

# y_true==0일 때 양수 예측(overshoot)에 대한 별도 패널티(작게라도 양수 찍는 습성 억제)
ZERO_OVERSHOOT_PENALTY = 0.25   # λ_zero (0.15~0.5 권장 범위)

# EarlyStopping 공통 설정(이미 쓰셨다면 그대로 두셔도 됩니다)
EARLY_STOP_PATIENCE = 6  # CV/Final 모두 동일하게 사용

# 추론 단계(리더보드 직결) 안전장치
USE_INT_ROUND      = False   # 제출이 정수 필수 아니라고 하셨으므로 기본 False 권장
CUT_THRESHOLD      = None    # 이하면 0으로 컷(0.7~1.0 사이 탐색)
ZERO_RUN_GUARD_DAYS= 14      # 직전 K일 합이 0이면 미래 7일 전부 0 강제

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

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # 원본 보존
    df["weekday"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["is_ski_season"] = df["month"].isin([12, 1, 2]).astype(int)

    years = sorted(df["date"].dt.year.unique().tolist())
    kr = set(country_holidays("KR", years=years))  # membership 검사 빠르게
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in kr)).astype(int)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["month_sin"]   = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"]   = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
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
    """모델 예측 yhat(7,)과 '최근 7일 그대로' 나이브를 섞는다."""
    a = ENSEMBLE_NAIVE_W if alpha is None else float(alpha)
    naive = g["sales"].astype(float).tail(PRED_LEN).to_numpy()
    if naive.shape[0] < PRED_LEN:
        naive = np.pad(naive, (PRED_LEN - naive.shape[0], 0), constant_values=0.0)
    return (1.0 - a) * yhat + a * naive


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
KNOWN_REAL_COLS = [
    "is_holiday", "is_weekend", "is_ski_season",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
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

    CTX = int(context_len) if context_len is not None else int(CONTEXT_LEN)
    PRED = int(prediction_len) if prediction_len is not None else int(PRED_LEN)

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
    else:
        for alt in ["control_columns", "conditional_columns", "categorical_columns"]:
            if alt in params:
                kwargs[alt] = known_real_cols
                break

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

def _log_mae_torch(y_true_log, y_pred_log):
    # 로그 공간 MAE: 저수량/제로 근처에서 안정화
    return torch.abs(y_pred_log - y_true_log)

def _zero_overshoot_penalty_torch(y_true, y_pred):
    # y_true==0에서 양수 예측 자체에 선형 패널티 (너무 강하지 않게 평균)
    mask = (y_true <= 1e-9).float()
    return (mask * y_pred.clamp(min=0.0)).mean()

def _mae_torch(y_true, y_pred):
    return torch.abs(y_pred - y_true)

def _reduce_mean(x):
    return torch.mean(x)

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
        self.w_zero = 1.10  # ← 1.25 에서 낮춤 (0 쏠림 약화)
        self.w_all0w = 1.00
        self.w_logmae = 0.03  # ← 0.30 → 0.03 (로그 MAE 과대 억제)
        self.w_ovr = 0.02  # ← 0.25 → 0.02 (0에서 양수 예측 과벌점 제거)
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
            try: pred = pred[:, self.target_ch, :]
            except Exception: pass

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
                sid = scf.squeeze(-1) if scf.dim() == 2 else scf  # (B,)
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
            if w["w_smape"]>0: loss += w["w_smape"] * torch.sum(W*smape) / (W.sum()+eps)
            if w["w_mae"]  >0: loss += w["w_mae"]   * torch.sum(W*mae)   / (W.sum()+eps)
            loss += self.w_logmae * torch.sum(W*log_mae) / (W.sum()+eps)
            loss += self.w_ovr    * torch.sum(W*overshot) / (W.sum()+eps)

            out.loss = loss

        if pred is not None:
            out.prediction = pred
        return out

def make_model():
    config = PatchTSTConfig(
        num_input_channels=1 + len(KNOWN_REAL_COLS),
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        patch_length=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
        d_model=256,
        num_attention_heads=16,
        num_hidden_layers=4,
        ffn_dim=512,
        dropout=0.10,
        head_dropout=0.10,
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

le  = fit_or_load_label_encoder(raw["store_menu"])
feat = add_time_features(raw)
df   = finalize_columns(feat, le)

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
fold_weeks = contiguous_week_folds(all_weeks_sorted, K_FOLDS)

cv_metrics = []
for fold, v_weeks in enumerate(fold_weeks):
    tr_m, va_m = make_masks_by_weeks(v_weeks, all_weeks_sorted, purge_gap=PURGE_GAP_WEEKS)
    train_df = df.loc[tr_m].copy()
    valid_df = df.loc[va_m].copy()

    train_ds = build_dataset(train_df)
    valid_ds = build_dataset(valid_df)

    model = make_model()
    # ADD ↓ 특수 매장 가중 사용
    model.special_store_ids = SPECIAL_STORE_IDS

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

    print(f"\n[CV] fold={fold} train_rows={len(train_df)} valid_rows={len(valid_df)} weeks={min(v_weeks)}..{max(v_weeks)}")
    trainer.train()
    trainer.eval_dataset = valid_ds
    m = trainer.evaluate()
    fold_dir = os.path.join(SAVE_DIR, f"fold_{fold}")
    trainer.save_model(fold_dir)
    m["fold"] = fold
    cv_metrics.append(m)
    print(f"[CV] fold={fold} metrics={m}")

cv_eval_loss = float(np.mean([m["eval_loss"] for m in cv_metrics]))
cv_smape = float(np.mean([m.get("eval_smape", np.nan) for m in cv_metrics]))
cv_mae   = float(np.mean([m.get("eval_mae",   np.nan) for m in cv_metrics]))
cv_rmse  = float(np.mean([m.get("eval_rmse",  np.nan) for m in cv_metrics]))
print(f"CV avg → loss={cv_eval_loss:.6f}, smape={cv_smape:.6f}, mae={cv_mae:.3f}, rmse={cv_rmse:.3f}")

# --- 폴드별 sMAPE 기반 가중치 (작을수록 가중↑) ---
fold_smapes = [m.get("eval_smape", np.inf) for m in cv_metrics]
if all(np.isfinite(s) for s in fold_smapes) and len(fold_smapes) > 0:
    w = 1.0 / (np.asarray(fold_smapes) + 1e-6)
    FOLD_WEIGHTS = (w / w.sum()).astype(float).tolist()
else:
    FOLD_WEIGHTS = [1.0 / max(1, len(cv_metrics))] * max(1, len(cv_metrics))
print("[FOLD ENSEMBLE] weights:", FOLD_WEIGHTS)

# ----------  # <CELL: final fit (all data)>
train_all = build_dataset(df)
valid_all = build_dataset(df)

final_model = make_model()
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
print("FINAL full-data eval:", final_metrics)

final_trainer.save_model(os.path.join(SAVE_DIR, "best"))
print("Saved:", os.path.join(SAVE_DIR, "best"))
print("LabelEncoder:", LE_PATH)

# ----------  # <CELL: load fold models for inference ensemble>
INFER_TRAINERS = []
if FOLD_ENSEMBLE:
    INFER_TRAINERS = []
    for fold in range(K_FOLDS):
        fold_dir = os.path.join(SAVE_DIR, f"fold_{fold}")
        # 둘 중 존재하는 걸 로드
        bin_path = None
        for fn in ["pytorch_model.bin", "model.safetensors"]:
            p = os.path.join(fold_dir, fn)
            if os.path.exists(p):
                bin_path = p
                break

        if bin_path is not None:
            m = make_model()
            if bin_path.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load

                sd = safe_load(bin_path)
            else:
                sd = torch.load(bin_path, map_location="cpu")

            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing:
                print(f"[fold {fold}] missing keys:", len(missing))
            if unexpected:
                print(f"[fold {fold}] unexpected keys:", len(unexpected))

            t = Trainer(
                model=m,
                args=training_args,
                data_collator=ts_data_collator,
                compute_metrics=compute_metrics,
            )
            INFER_TRAINERS.append(t)
        else:
            print(f"[FOLD ENSEMBLE] checkpoint not found in {fold_dir}")

    print(f"[FOLD ENSEMBLE] loaded {len(INFER_TRAINERS)} fold models.")
else:
    INFER_TRAINERS = []

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

def _zero_run_guard(g: pd.DataFrame, yhat: np.ndarray) -> np.ndarray:
    v = pd.to_numeric(g["매출수량"], errors="coerce").fillna(0).to_numpy()
    if ZERO_RUN_GUARD_DAYS > 0 and len(v) >= ZERO_RUN_GUARD_DAYS:
        if v[-ZERO_RUN_GUARD_DAYS:].sum() == 0:
            return np.zeros_like(yhat, dtype=float)
    return yhat

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
        "매출수량": 0.0,
        "store_name": store,
        "menu_name": menu,
        "store_menu": store_menu,
        "date": future_dates,
    })
    fut = add_time_features(fut)
    return fut

def _predict_last_window_for_file(file_path, sample_df):
    """
    한 개 TEST_xx.csv에 대해:
      - 각 '영업장명_메뉴명'별 마지막 CTX일 + 미래 7일 구성
      - 마지막 윈도우 7일만 예측
      - baseline 포맷(영업일자='TEST_xx+{d}일')으로 레코드 반환
    """
    test_prefix = re.search(r"(TEST_\d+)", os.path.basename(file_path)).group(1)

    # 현재 로더의 모델 컨텍스트 길이를 직접 읽어 동기화
    CTX = int(getattr(final_trainer.model.base.config, "context_length", CONTEXT_LEN))

    df_t = pd.read_csv(file_path)
    df_t["date"] = pd.to_datetime(df_t["영업일자"])
    s = df_t["영업장명_메뉴명"].astype(str).str.split("_", n=1, expand=True)
    df_t["store_name"] = s[0]; df_t["menu_name"] = s[1]
    df_t["store_menu"] = df_t["store_name"] + "_" + df_t["menu_name"]
    df_t["매출수량"] = pd.to_numeric(df_t["매출수량"], errors="coerce").fillna(0.0)
    df_t.loc[df_t["매출수량"] < 0, "매출수량"] = 0.0
    df_t["sales"] = df_t["매출수량"].astype(float)  # 나이브 섞기 용
    # (옵션) 훈련과 동일 정규화를 원하면 주석 해제
    df_t = enforce_regular_daily(df_t)
    df_t = add_time_features(df_t)

    records = []
    for store_menu, g in df_t.groupby("영업장명_메뉴명"):
        g_raw = g.sort_values("date").copy()  # ← 원본(무패딩) 보관
        g = leftpad_to_context(g_raw, CTX, store_menu)  # 모델 입력용 패딩

        last_date = g["date"].max()
        fut = _make_future_rows(store_menu, last_date, horizon=PRED_LEN)
        combo = pd.concat([g, fut], ignore_index=True)
        combo["sales"] = pd.to_numeric(combo["매출수량"], errors="coerce").fillna(0.0)

        # 마지막 윈도우만 남김
        combo_tail = combo.iloc[-(CTX + PRED_LEN):].copy()

        # 라벨인코더(테스트 신규 ID 포함) — raw가 없을 수도 있으니 가드
        base_series = raw["store_menu"] if "raw" in globals() else combo_tail["store_menu"]
        le2 = fit_or_load_label_encoder(pd.concat([base_series, combo_tail["store_menu"]]))
        combo_fin = finalize_columns(combo_tail, le2)

        # 이 아이템만 있는 dataset 생성 → 정확히 1 샘플 (모델 CTX로 빌드)
        ds = build_dataset(combo_fin, context_len=CTX, prediction_len=PRED_LEN)

        # --- Fold ensemble predictors ---
        # 폴드 모델이 있으면 그것만 사용, 없으면 최종 모델 하나 사용
        predictors = INFER_TRAINERS if (
                    FOLD_ENSEMBLE and 'INFER_TRAINERS' in globals() and len(INFER_TRAINERS) > 0) else [final_trainer]

        yhat_list = []
        for t in predictors:
            # predict 순간만 워커/배치/퍼시스턴트 워커 설정을 임시로 조정
            old_workers = getattr(t.args, "dataloader_num_workers", None)
            old_eval_bs = getattr(t.args, "per_device_eval_batch_size", None)
            old_persist = getattr(t.args, "dataloader_persistent_workers", None)

            t.args.dataloader_num_workers = 0
            t.args.per_device_eval_batch_size = min(16, (old_eval_bs or 16))
            if old_persist is not None:
                t.args.dataloader_persistent_workers = False

            try:
                preds_out = t.predict(ds)
            finally:
                # 원복
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

        S = np.stack(yhat_list, axis=0)  # (n_models, 7)
        if S.shape[0] == 1:
            yhat = S[0]
        else:
            w = np.asarray(globals().get("FOLD_WEIGHTS", []), dtype=float)
            if w.size != S.shape[0] or not np.isfinite(w).all():
                print(f"[WARN] FOLD_WEIGHTS invalid (len={w.size}, models={S.shape[0]}) → uniform avg")
                w = np.ones(S.shape[0], dtype=float) / S.shape[0]
            yhat = np.average(S, axis=0, weights=w)

        if ENSEMBLE_NAIVE_W > 0:
            yhat = _blend_with_naive(yhat, g)

        # 제로-런 가드는 '실제 최근 K일' 기준으로만 판단
        yhat = _zero_run_guard(g_raw, yhat)
        cap = ITEM_CAP.get(store_menu, None)
        if cap is not None and np.isfinite(cap):
            yhat = np.minimum(yhat, float(cap) * float(CAP_MULT))

        # 출력 레코드 적재
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
        df_pred_one = _predict_last_window_for_file(p, submit_df)

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