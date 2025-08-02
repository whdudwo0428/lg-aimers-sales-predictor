"""
Exploratory Data Analysis for the Gonjiam resort demand forecasting challenge.

This script loads the training data, performs basic descriptive
statistics, and generates a handful of plots to aid understanding of
the data distribution and temporal patterns.  All plots are saved to
the `open/EDA/figures` directory for later inspection.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from util import add_date_features

plt.style.use('seaborn-v0_8-darkgrid')

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / 'dataset' / 'train' / 'train.csv'
    figures_dir = base_dir / 'EDA' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)
    # Add date features
    df = add_date_features(df, date_col='영업일자')

    # Descriptive statistics
    stats = df['매출수량'].describe()
    print("Sales quantity summary statistics:\n", stats)

    # Plot distribution of sales (log scale to handle skew)
    plt.figure(figsize=(8, 4))
    sns.histplot(df['매출수량'], bins=100, log_scale=(False, True))
    plt.title('Distribution of Sales Quantity (log y-scale)')
    plt.xlabel('Sales quantity')
    plt.ylabel('Frequency (log scale)')
    plt.tight_layout()
    plt.savefig(figures_dir / 'sales_distribution.png')
    plt.close()

    # # Top 10 menus by total sales
    # top_menus = df.groupby('영업장명_메뉴명')['매출수량'].sum().sort_values(ascending=False).head(10)
    # plt.figure(figsize=(10, 5))
    # sns.barplot(x=top_menus.values, y=top_menus.index)
    # plt.title('Top 10 Menus by Total Sales')
    # plt.xlabel('Total sales quantity')
    # plt.ylabel('Menu')
    # plt.tight_layout()
    # plt.savefig(figures_dir / 'top_menus.png')
    # plt.close()

    # Aggregate daily sales over all menus
    daily_sales = df.groupby('영업일자')['매출수량'].sum().reset_index()
    daily_sales['영업일자'] = pd.to_datetime(daily_sales['영업일자'])
    plt.figure(figsize=(12, 4))
    plt.plot(daily_sales['영업일자'], daily_sales['매출수량'])
    plt.title('Total Daily Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total sales quantity')
    plt.tight_layout()
    plt.savefig(figures_dir / 'total_daily_sales.png')
    plt.close()

    # Sales by day of week
    dow_sales = df.groupby('dayofweek')['매출수량'].mean()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=dow_sales.index, y=dow_sales.values)
    plt.title('Average Sales by Day of Week (0=Mon)')
    plt.xlabel('Day of week')
    plt.ylabel('Average sales quantity')
    plt.tight_layout()
    plt.savefig(figures_dir / 'sales_by_dow.png')
    plt.close()

    # Sales by month
    month_sales = df.groupby('month')['매출수량'].mean()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=month_sales.index, y=month_sales.values)
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average sales quantity')
    plt.tight_layout()
    plt.savefig(figures_dir / 'sales_by_month.png')
    plt.close()

    print(f"EDA figures saved to {figures_dir}")

if __name__ == "__main__":
    main()
