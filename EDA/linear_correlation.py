"""
Exploratory Data Analysis for the Gonjiam resort demand forecasting challenge.

This script loads the training data, performs basic descriptive
statistics, and generates a handful of plots to aid understanding of
the data distribution and temporal patterns.  All plots are saved to
the `open/EDA/figures` directory for later inspection.
"""
import os
import sys
from pathlib import Path

import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# sys.path에 추가
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src_FEDmodel.data_module import add_date_features, add_store_menu_features

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / 'dataset' / 'train' / 'train.csv'
    figures_dir = base_dir / 'EDA' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)

    # 1. 메뉴들간 상관관계
    df1 = df.pivot(columns='영업장명_메뉴명', index='영업일자', values='매출수량').reset_index()
    menu_cols = df1.columns.drop('영업일자')
    corr_matrix1 = df1[menu_cols].corr()

    corr_matrix1.index.name = None
    corr_matrix1.columns.name = None

    corr_pairs1 = (
        corr_matrix1.unstack()
        .reset_index()
        .rename(columns={'level_0': '메뉴1', 'level_1': '메뉴2', 0: '상관계수'})
    )
    corr_pairs1 = corr_pairs1[corr_pairs1['메뉴1'] < corr_pairs1['메뉴2']]

    # 절댓값 기준으로 상위 20쌍 추출
    top_corr1 = corr_pairs1.reindex(corr_pairs1['상관계수'].abs().sort_values(ascending=False).index)

    print("\n메뉴간의 상관계수 상위 20쌍:")
    print(top_corr1.head(20))

    # 2. 날짜 변수와 메뉴들간 상관관계
    df2 = add_date_features(df1)
    date_cols = [c for c in df2.columns if c not in df1.columns and c != '영업일자']

    combined_cols = list(menu_cols) + date_cols
    corr_matrix2 = df2[combined_cols].corr()

    corr_matrix2.index.name = None
    corr_matrix2.columns.name = None

    # 날짜 변수(행) x 메뉴 변수(열) 구간만 뽑고 절댓값 취함
    corr_pairs2 = (
        corr_matrix2.loc[menu_cols, date_cols]
        .abs()
        .unstack()
        .reset_index()
    )
    corr_pairs2.columns = ['날짜변수', '메뉴', '상관계수']

    # 절댓값 기준 상위 20쌍 추출
    top_corr2 = corr_pairs2.sort_values(by='상관계수', ascending=False)

    print("\n날짜 파생변수와 메뉴 간 상관계수 상위 20쌍:")
    print(top_corr2.head(20))

    # 3. 결과 CSV 저장
    save_dir = base_dir / 'EDA' / 'correlation'
    save_dir.mkdir(parents=True, exist_ok=True)

    top_corr1.to_csv(save_dir / 'linear_menu_correlation.csv', index=False, encoding='utf-8-sig')
    top_corr2.to_csv(save_dir / 'linear_date_menu_correlation.csv', index=False, encoding='utf-8-sig')

    print(f"\n상관계수 결과가 '{save_dir}' 폴더에 CSV로 저장되었습니다.")

if __name__ == "__main__":
    main()
