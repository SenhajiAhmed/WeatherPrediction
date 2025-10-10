#!/usr/bin/env python3
"""
02_feature_engineering_all.py

Applies the daily aggregation and next-day target generation pipeline
to ALL cleaned monthly CSV files inside 'era5_data_csv_cleaned/'.

Each step produces:
  - *_daily_features.csv   (daily aggregated + next-day targets)

Run:
  python 02_feature_engineering_all.py
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FOLDER = "era5_data_csv_cleaned"
OUTPUT_FOLDER = "era5_features"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
np.random.seed(42)

# --- Main loop over all cleaned CSVs ---
for file in sorted(os.listdir(INPUT_FOLDER)):
    if not file.endswith("_cleaned.csv"):
        continue

    in_path = os.path.join(INPUT_FOLDER, file)
    base_name = os.path.splitext(file)[0].replace("_cleaned", "")
    print(f"\n🚀 Processing {file}")

    # === Load cleaned data ===
    df = pd.read_csv(in_path, parse_dates=['valid_time'])
    df = df.sort_values(['longitude', 'latitude', 'valid_time']).reset_index(drop=True)
    print(f"Loaded: {df.shape} | Date range: {df['valid_time'].min()} → {df['valid_time'].max()}")

    # === Create 'date' column ===
    df['date'] = df['valid_time'].dt.date

    # === Identify weather parameters to aggregate ===
    exclude_cols = ['valid_time', 'date', 'longitude', 'latitude', 'number', 'expver']
    weather_params = [col for col in df.columns if col not in exclude_cols]
    print(f"Weather parameters: {weather_params}")

    # === Aggregate to daily min/max/mean ===
    agg_dict = {param: ['min', 'max', 'mean'] for param in weather_params}
    daily_stats = df.groupby(['date', 'latitude', 'longitude']).agg(agg_dict).reset_index()
    daily_stats.columns = ['date', 'latitude', 'longitude'] + [
        f"{param}_{stat}" for param in weather_params for stat in ['min', 'max', 'mean']
    ]
    print(f"Daily aggregation done: {daily_stats.shape}")

    # === Create next-day targets ===
    daily_stats = daily_stats.sort_values(['latitude', 'longitude', 'date']).reset_index(drop=True)
    variables_to_predict = ['t2m']
    target_cols = []
    for var in variables_to_predict:
        for stat in ['min', 'max', 'mean']:
            col = f"{var}_{stat}"
            next_col = f"{col}_next"
            daily_stats[next_col] = daily_stats.groupby(['latitude', 'longitude'])[col].shift(-1)
            target_cols.append(next_col)
    print(f"Created targets: {target_cols}")

    # Drop incomplete rows (last day per location)
    daily_stats_complete = daily_stats.dropna(subset=target_cols).reset_index(drop=True)
    print(f"After dropna: {daily_stats_complete.shape}")

    # Save daily features file only
    daily_out = os.path.join(OUTPUT_FOLDER, f"{base_name}_daily_features.csv")
    daily_stats_complete.to_csv(daily_out, index=False)
    print(f"✓ Saved daily features → {daily_out}")

print("\n🎉 All monthly files processed successfully (daily features only)!")
