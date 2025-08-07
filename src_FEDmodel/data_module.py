import torch
import pandas as pd
import numpy as np
import datetime
from typing import List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Predefined set of major Korean public holidays for 2023–2025.  These dates
# reflect official public holidays and some substitution days that fall
# within the training and evaluation horizon (2023-01-01 through
# 2025-05-24).  Only holidays up to May 2025 are included, as the
# test period ends in May 2025.
KOREAN_HOLIDAYS: List[datetime.date] = [
    # 2023 holidays
    datetime.date(2023, 1, 1),  # New Year's Day
    datetime.date(2023, 1, 21), datetime.date(2023, 1, 22), datetime.date(2023, 1, 23),
    datetime.date(2023, 1, 24),  # Seollal (Korean New Year) + substitute holiday
    datetime.date(2023, 3, 1),   # Independence Movement Day
    datetime.date(2023, 5, 5),   # Children's Day
    datetime.date(2023, 5, 27),  # Buddha's Birthday
    datetime.date(2023, 6, 6),   # Memorial Day
    datetime.date(2023, 8, 15),  # Liberation Day
    datetime.date(2023, 9, 28), datetime.date(2023, 9, 29), datetime.date(2023, 9, 30),
    datetime.date(2023, 10, 2),  # Chuseok holidays & substitute
    datetime.date(2023, 10, 3),  # National Foundation Day
    datetime.date(2023, 10, 9),  # Hangul Proclamation Day
    datetime.date(2023, 12, 25), # Christmas Day
    # 2024 holidays
    datetime.date(2024, 1, 1),
    datetime.date(2024, 2, 9), datetime.date(2024, 2, 10), datetime.date(2024, 2, 11),
    datetime.date(2024, 2, 12),  # Seollal & substitute holiday
    datetime.date(2024, 3, 1),   # Independence Movement Day
    datetime.date(2024, 4, 10),  # National Assembly Election Day
    datetime.date(2024, 5, 5), datetime.date(2024, 5, 6),  # Children's Day + substitute
    datetime.date(2024, 5, 15),  # Buddha's Birthday
    datetime.date(2024, 6, 6),   # Memorial Day
    datetime.date(2024, 8, 15),  # Liberation Day
    datetime.date(2024, 9, 16), datetime.date(2024, 9, 17), datetime.date(2024, 9, 18),
    datetime.date(2024, 10, 3),  # National Foundation Day
    datetime.date(2024, 10, 9),  # Hangul Proclamation Day
    datetime.date(2024, 12, 25), # Christmas Day
    # 2025 holidays (until May)
    datetime.date(2025, 1, 1),   # New Year's Day
    datetime.date(2025, 1, 28), datetime.date(2025, 1, 29), datetime.date(2025, 1, 30),  # Seollal Holiday
    datetime.date(2025, 3, 3),   # March 1st substitute
    datetime.date(2025, 5, 5),   # Children's Day
    datetime.date(2025, 5, 6),   # Buddha's Birthday substitute
    datetime.date(2025, 6, 6),   # Memorial Day
    datetime.date(2025, 8, 15),  # Liberation Day
    datetime.date(2025, 10, 3),  # National Foundation Day
    datetime.date(2025, 10, 6), datetime.date(2025, 10, 7), datetime.date(2025, 10, 8),  # Chuseok Holiday
    datetime.date(2025, 10, 9),  # Hangul Day
    datetime.date(2025, 12, 25), # Christmas
]
# 연휴일자에 따라 가중치 추가

def add_date_features(df: pd.DataFrame, date_col: str = '영업일자') -> pd.DataFrame:
    """Augment the DataFrame with additional calendar-based features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the date column.
    date_col : str, optional
        Name of the date column to process. Defaults to '영업일자'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with added feature columns.
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    dt_series = result[date_col]
    result['year'] = dt_series.dt.year
    result['month'] = dt_series.dt.month
    result['day'] = dt_series.dt.day
    result['dayofweek'] = dt_series.dt.dayofweek  # Monday=0
    result['weekofyear'] = dt_series.dt.isocalendar().week.astype(int)
    result['is_weekend'] = result['dayofweek'].isin([5, 6]).astype(int)
    # Holiday flags
    date_list = dt_series.dt.date
    result['is_holiday'] = date_list.isin(KOREAN_HOLIDAYS).astype(int)
    # Before/after holiday flags
    holiday_set = set(KOREAN_HOLIDAYS)
    before_holiday = set(d - datetime.timedelta(days=1) for d in holiday_set)
    after_holiday = set(d + datetime.timedelta(days=1) for d in holiday_set)
    result['is_holiday_eve'] = date_list.isin(before_holiday).astype(int)
    result['is_holiday_after'] = date_list.isin(after_holiday).astype(int)
    # Quarter and day of year
    result['quarter'] = dt_series.dt.quarter
    result['dayofyear'] = dt_series.dt.dayofyear
    # Season (1: Spring, 2: Summer, 3: Autumn, 4: Winter)
    month = result['month']
    result['season'] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [12, 1, 2],
        default=4
    )
    # Ski season (1 if date falls in ski season; December–February)
    result['ski_season'] = month.isin([12, 1, 2]).astype(int)
    return result

def add_store_menu_features(df: pd.DataFrame, col_name: str = '영업장명_메뉴명') -> pd.DataFrame:
    """
    '영업장명_메뉴명' 컬럼을 '_'로 분리해
    '영업장명'과 '메뉴명' 컬럼을 생성합니다.
    """
    result = df.copy()
    split_cols = result[col_name].str.split('_', n=1, expand=True)
    result['영업장명'] = split_cols[0]
    result['메뉴명'] = split_cols[1]
    return result

# 제공해주신 add_store_menu_features 함수는 현재의 FEDformer 모델 구조에서는 사용하지 않습니다.
# 이유:
# 현재 TimeSeriesDataModule은 pivot_table을 사용하여 '영업장명_메뉴명'의 각 고유값을 하나의 독립된 시계열 특성(컬럼)으로 변환합니다.
# FEDformer와 같은 다변량 시계열 모델은 이런 데이터 구조를 사용합니다.
# add_store_menu_features 함수는 데이터를 피벗하지 않고 '영업장명'과 '메뉴명'을 별도의 범주형 변수로 사용하는 LightGBM이나 XGBoost 같은 트리 기반 모델에 더 적합한 전처리 방식입니다.

class FeatureEngineer:
    def __init__(self, lag_periods=None, ma_windows=None):
        """
        Lag와 이동 평균 피처 생성만 담당하도록 수정.
        """
        self.lag_periods = lag_periods if lag_periods is not None else []
        self.ma_windows = ma_windows if ma_windows is not None else []

    def _create_lag_features(self, df):
        if not self.lag_periods: return []
        return [df.shift(lag).rename(columns=lambda c: f"{c}_lag_{lag}") for lag in self.lag_periods]

    def _create_ma_features(self, df):
        if not self.ma_windows: return []
        return [df.rolling(window=w, min_periods=1).mean().rename(columns=lambda c: f"{c}_ma_{w}") for w in self.ma_windows]

    def transform(self, pivot_df):
        """
        입력된 피벗 테이블에 Lag와 MA 피처를 생성하여 데이터프레임 리스트로 반환합니다.
        """
        features_to_concat = []
        features_to_concat.extend(self._create_lag_features(pivot_df))
        features_to_concat.extend(self._create_ma_features(pivot_df))
        
        if not features_to_concat:
            return None
        
        # 생성된 피처들을 하나의 데이터프레임으로 합침
        return pd.concat(features_to_concat, axis=1)


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str, sequence_length: int, forecast_horizon: int, 
                 label_len: int, batch_size: int, feature_engineer: FeatureEngineer = None):
        
        super().__init__()
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.label_len = label_len
        self.batch_size = batch_size
        self.feature_engineer = feature_engineer
        self.data = None
        self.target_columns = None
        self.time_feature_columns = None
    
    @property
    def input_dim(self):
        if self.data is None:
            self.prepare_data()
        return self.data.shape[1]
        
    @property
    def output_dim(self):
        if self.target_columns is None:
            self.prepare_data()
        return len(self.target_columns)
        

    def prepare_data(self):
        if self.data is not None: return

        df = pd.read_csv(self.file_path)
        df['영업일자'] = pd.to_datetime(df['영업일자'])
        pivot_df = df.pivot_table(index='영업일자', columns='영업장명_메뉴명', values='매출수량').fillna(0)
        
        data_parts = [pivot_df]
        
        if self.feature_engineer:
            additional_features = self.feature_engineer.transform(pivot_df)
            if additional_features is not None:
                data_parts.append(additional_features)
        
        # --- [수정] 새로운 add_date_features 함수를 사용하여 풍부한 시간 피처 생성 ---
        # 1. pivot_df의 인덱스(날짜)를 사용하여 임시 데이터프레임 생성
        date_df = pd.DataFrame({'영업일자': pivot_df.index})
        
        # 2. add_date_features 함수 적용
        time_features = add_date_features(date_df, date_col='영업일자')
        
        # 3. 다른 데이터와 합치기 위해 인덱스 설정
        time_features.set_index('영업일자', inplace=True)
        
        # 4. 생성된 시간 피처들의 컬럼명을 저장
        self.time_feature_columns = time_features.columns
        data_parts.append(time_features)
        # --- 수정 끝 ---

        self.data = pd.concat(data_parts, axis=1).astype(np.float32)
        self.data.fillna(0, inplace=True)
        
        self.target_columns = pivot_df.columns
        print("FeatureEngineer 및 신규 날짜 피처 사용 활성화. 최종 데이터 모양:", self.data.shape)

    def preprocess_inference_data(self, df: pd.DataFrame):
        pivot_df = df.pivot_table(index='영업일자', columns='영업장명_메뉴명', values='매출수량').fillna(0)
        pivot_df = pivot_df.reindex(columns=self.target_columns, fill_value=0)

        data_parts = [pivot_df]
        if self.feature_engineer:
            additional_features = self.feature_engineer.transform(pivot_df)
            if additional_features is not None:
                data_parts.append(additional_features)
        
        # --- [수정] 추론 데이터에도 동일한 날짜 피처 생성 로직 적용 ---
        date_df = pd.DataFrame({'영업일자': pivot_df.index})
        time_features = add_date_features(date_df, date_col='영업일자')
        time_features.set_index('영업일자', inplace=True)
        data_parts.append(time_features)
        # --- 수정 끝 ---

        final_df = pd.concat(data_parts, axis=1).astype(np.float32)
        final_df.fillna(0, inplace=True)
        
        return final_df
    
    def create_sequences(self, data):
        # 전체 시퀀스 길이는 인코더 입력(seq_len) + 예측 구간(horizon)
        total_seq_len = self.sequence_length + self.forecast_horizon
        
        x_values, y_values, x_mark_values, y_mark_values = [], [], [], []

        for i in range(len(data) - total_seq_len + 1):
            # 1. 인코더 입력 (batch_x)
            x_seq = data.iloc[i : i + self.sequence_length][self.target_columns].values
            x_values.append(x_seq)

            # 2. 인코더 시간 특징 (batch_x_mark)
            x_mark_seq = data.iloc[i : i + self.sequence_length][self.time_feature_columns].values
            x_mark_values.append(x_mark_seq)

            # 3. 디코더 입력 (batch_y)
            # 디코더 입력은 "힌트" 부분(label_len)과 실제 예측해야 할 부분(horizon)으로 구성
            y_seq_start = i + self.sequence_length - self.label_len
            y_seq_end = i + total_seq_len
            y_seq = data.iloc[y_seq_start:y_seq_end][self.target_columns].values
            y_values.append(y_seq)
            
            # 4. 디코더 시간 특징 (batch_y_mark)
            y_mark_seq = data.iloc[y_seq_start:y_seq_end][self.time_feature_columns].values
            y_mark_values.append(y_mark_seq)

        # 모든 데이터를 numpy 배열로 변환
        return (np.array(x_values), np.array(y_values), 
                np.array(x_mark_values), np.array(y_mark_values))

    def setup(self, stage: str):
        if self.data is None: self.prepare_data()
            
        total_len = len(self.data)
        train_size = int(total_len * 0.7)
        val_size = int(total_len * 0.15)
        
        train_df = self.data.iloc[:train_size]
        val_df = self.data.iloc[train_size:train_size + val_size]
        test_df = self.data.iloc[train_size + val_size:]

        if stage == 'fit' or stage is None:
            self.train_dataset = self.create_sequences(train_df)
            self.val_dataset = self.create_sequences(val_df)

        if stage == 'test' or stage is None:
            self.test_dataset = self.create_sequences(test_df)

    # --- FIX: DataLoader가 여러 개의 텐서를 반환하도록 수정 ---
    def _create_dataloader(self, sequence_data, shuffle=False):
        # numpy 배열들을 torch 텐서로 변환
        tensors = [torch.from_numpy(d) for d in sequence_data]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)