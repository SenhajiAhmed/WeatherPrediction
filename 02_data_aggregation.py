#!/usr/bin/env python3
"""
02_feature_engineering_all.py

Applies the daily aggregation and 7-day window feature construction pipeline
to ALL cleaned monthly CSV files inside 'era5_data_csv_cleaned/'.

Each step produces:
  - *_daily_features.csv   (daily aggregated + next-day targets)
  - *_window_features.csv  (aggregated rolling window features ready for ML)

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
WINDOW_SIZE = 7
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
np.random.seed(42)

# --- Helper: Rolling-window feature construction ---
def create_window_features_aggregated(df, window_size, feature_cols, target_cols):
    """
    Create aggregated statistical features from windows.
    For each feature, compute: mean, std, min, max, last, first, trend, recent_3d.
    """
    features_list = []
    targets_list = []
    
    for (lat, lon), group in tqdm(df.groupby(['latitude', 'longitude'], sort=False),
                                   desc="Processing locations (aggregated)"):
        group = group.sort_values('date').reset_index(drop=True)
        if len(group) <= window_size:
            continue
        
        for i in range(window_size, len(group)):
            window = group.iloc[i - window_size:i]
            feature_row = {
                'date': group['date'].iloc[i],
                'latitude': lat,
                'longitude': lon,
            }
            
            # Compute statistics for each feature
            for col in feature_cols:
                window_vals = window[col].values
                feature_row[f"{col}_mean"] = np.mean(window_vals)
                feature_row[f"{col}_std"] = np.std(window_vals)
                feature_row[f"{col}_min"] = np.min(window_vals)
                feature_row[f"{col}_max"] = np.max(window_vals)
                feature_row[f"{col}_last"] = window_vals[-1]
                feature_row[f"{col}_first"] = window_vals[0]
                feature_row[f"{col}_trend"] = window_vals[-1] - window_vals[0]
                if window_size >= 3:
                    feature_row[f"{col}_recent_3d"] = np.mean(window_vals[-3:])
            
            y_target = group[target_cols].iloc[i].values
            if not np.any(np.isnan(y_target)):
                features_list.append(feature_row)
                targets_list.append(y_target)
    
    X = pd.DataFrame(features_list)
    y = pd.DataFrame(targets_list, columns=target_cols)
    return X, y

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

    # Save daily features file
    daily_out = os.path.join(OUTPUT_FOLDER, f"{base_name}_daily_features.csv")
    daily_stats_complete.to_csv(daily_out, index=False)
    print(f"✓ Saved daily features → {daily_out}")

    # === 7-day window feature creation ===
    exclude_cols = target_cols + ['date', 'latitude', 'longitude']
    feature_cols = [c for c in daily_stats_complete.columns if c not in exclude_cols]
    print(f"Feature cols: {len(feature_cols)}, Target cols: {len(target_cols)}")

    print(f"Building 7-day aggregated window features...")
    X_agg, y_agg = create_window_features_aggregated(
        daily_stats_complete, WINDOW_SIZE, feature_cols, target_cols
    )

    print(f"✅ Window features created: {len(X_agg)} samples")
    out_X = os.path.join(OUTPUT_FOLDER, f"{base_name}_window_features.csv")
    merged = pd.concat([X_agg, y_agg], axis=1)
    merged.to_csv(out_X, index=False)
    print(f"✓ Saved window features → {out_X}")

print("\n🎉 All monthly files processed successfully!")
