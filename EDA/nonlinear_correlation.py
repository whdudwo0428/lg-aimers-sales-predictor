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
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# sys.path에 추가
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src_FEDmodel.data_module import add_date_features, add_store_menu_features

def calc_spearman_corr_matrix(df):
    cols = df.columns
    n = len(cols)
    corr_mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    for i in range(n):
        for j in range(i, n):
            corr, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            corr_mat.iloc[i, j] = corr
            corr_mat.iloc[j, i] = corr
    return corr_mat

def main():
    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / 'dataset' / 'train' / 'train.csv'
    figures_dir = base_dir / 'EDA' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)

    # 1. 메뉴들간 상관관계 (스피어만, 거리)
    df1 = df.pivot(columns='영업장명_메뉴명', index='영업일자', values='매출수량').reset_index()
    menu_cols = df1.columns.drop('영업일자')
    df_menu = df1[menu_cols]

    print("Calculating Spearman correlation matrix for menus...")
    spearman_corr_menu = calc_spearman_corr_matrix(df_menu)

    spearman_corr_menu.index.name = None
    spearman_corr_menu.columns.name = None


    # Spearman 상관계수 쌍 만들기
    spearman_pairs = (
        spearman_corr_menu.unstack()
        .reset_index()
        .rename(columns={'level_0': '메뉴1', 'level_1': '메뉴2', 0: 'Spearman_상관계수'})
    )
    spearman_pairs = spearman_pairs[spearman_pairs['메뉴1'] < spearman_pairs['메뉴2']]

    # 스피어만 기준 상위
    top_spearman = spearman_pairs.reindex(spearman_pairs['Spearman_상관계수'].abs().sort_values(ascending=False).index)

    print("\n스피어만 상관계수 기준 메뉴간 상위 20쌍:")
    print(top_spearman.head(20))

    # 2. 날짜 변수와 메뉴들간 상관관계 (스피어만)
    df2 = add_date_features(df1)
    date_cols = [c for c in df2.columns if c not in df1.columns and c != '영업일자']
    combined_cols = list(menu_cols) + date_cols
    df_combined = df2[combined_cols]

    print("Calculating Spearman correlation matrix for date variables and menus...")
    spearman_corr_all = calc_spearman_corr_matrix(df_combined)

    spearman_corr_all.index.name = None
    spearman_corr_all.columns.name = None

    # 날짜변수 x 메뉴 변수 구간
    spearman_date_menu = (
        spearman_corr_all.loc[date_cols, menu_cols]
        .abs()
        .unstack()
        .reset_index()
    )
    spearman_date_menu.columns = ['날짜변수', '메뉴', 'Spearman_상관계수']
    top_date_menu_spearman = spearman_date_menu.sort_values(by='Spearman_상관계수', ascending=False)

    print("\n스피어만 상관계수 기준 날짜변수와 메뉴 간 상위 20쌍:")
    print(top_date_menu_spearman.head(20))

    # CSV 저장
    save_dir = base_dir / 'EDA' / 'correlation'
    save_dir.mkdir(parents=True, exist_ok=True)

    top_spearman.to_csv(save_dir / 'nonlinear_menu_spearman_correlation.csv', index=False, encoding='utf-8-sig')
    top_date_menu_spearman.to_csv(save_dir / 'nonlinear_date_menu_spearman_correlation.csv', index=False, encoding='utf-8-sig')

    print(f"\n상관계수 결과가 '{save_dir}' 폴더에 CSV로 저장되었습니다.")
    
if __name__ == "__main__":
    main()
