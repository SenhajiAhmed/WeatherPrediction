#!/usr/bin/env python3
"""
01_data_cleaning_all.py

Applies the cleaning process from 01_data_cleaning.ipynb to *all* CSV files in a folder.
Each cleaned CSV is saved to 'era5_data_csv_cleaned/' with '_cleaned' added to the filename.

Run: python 01_data_cleaning_all.py
"""

import pandas as pd
import numpy as np
import os

# --- Configuration ---
input_folder = "era5_data_csv"
output_folder = "era5_data_csv_cleaned"
os.makedirs(output_folder, exist_ok=True)

# --- Helper function to report changes ---
def report_changes(df, col, mask, action_desc):
    n_changed = mask.sum()
    print(f"{action_desc}: {n_changed} values in '{col}' were modified or removed.")
    return n_changed

# --- Processing loop ---
for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue

    csv_path = os.path.join(input_folder, filename)
    print(f"\n🚀 Processing: {filename}")

    # --- Step 1: Load CSV ---
    df = pd.read_csv(csv_path, parse_dates=['valid_time'])
    print('Step 1 complete: CSV loaded')

    # --- Step 2: Basic inspection ---
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # --- Step 3: Set time index ---
    df = df.sort_values('valid_time')
    df.set_index('valid_time', inplace=True)
    print('Step 3 complete: valid_time is now DatetimeIndex')

    # --- Step 4: Unit conversion ---
    for temp_col in ['d2m', 'skt']:
        if temp_col in df.columns:
            df[temp_col] = df[temp_col] - 273.15

    if 'msl' in df.columns:
        df['msl'] = df['msl'] / 100

    print('✅ Step 4 complete: units converted where applicable')

    # --- Step 5: Handle missing values ---
    print("Missing values before interpolation:")
    print(df.isna().sum())

    df.interpolate(method='time', limit_direction='both', inplace=True)
    df.dropna(inplace=True)

    print(f"Step 5 complete: missing values handled (shape now {df.shape})")

    # --- Step 6: Outliers / sanity checks ---
    # Temperature sanity range (°C)
    for temp_col, raw_col in zip(
        ['2m_temperature', '2m_dewpoint_temperature', 'skin_temperature'],
        ['t2m', 'd2m', 'skt']
    ):
        if raw_col in df.columns:
            mask = (df[raw_col] < -123.15) | (df[raw_col] > 57.0)
            report_changes(df, raw_col, mask, "Out-of-range temperature values removed")
            df = df[(df[raw_col] >= -123.15) & (df[raw_col] <= 57.0)]

    # Cloud cover: clip 0–1
    if 'tcc' in df.columns:
        before_clip = df['tcc'].copy()
        df['tcc'] = df['tcc'].clip(0, 1)
        changed = (before_clip != df['tcc']).sum()
        print(f"Cloud cover clipped (0–1): {changed} values adjusted.")

    # Precipitation: cannot be negative
    if 'tp' in df.columns:
        before_clip = df['tp'].copy()
        df['tp'] = df['tp'].clip(lower=0)
        changed = (before_clip != df['tp']).sum()
        print(f"Precipitation negatives removed: {changed} values adjusted.")

    # Wind components: cap ±100 m/s
    for wind_col in ['u10', 'v10']:
        if wind_col in df.columns:
            before_clip = df[wind_col].copy()
            df[wind_col] = df[wind_col].clip(-100, 100)
            changed = (before_clip != df[wind_col]).sum()
            print(f"Wind component '{wind_col}' clipped to ±100 m/s: {changed} values adjusted.")

    print("✅ Step 6 complete: outliers handled")

    # --- Step 7: Save cleaned CSV ---
    df.reset_index(inplace=True)
    base_name = os.path.splitext(filename)[0]
    cleaned_path = os.path.join(output_folder, f"{base_name}_cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    print(f"✅ Saved cleaned file: {cleaned_path}")
    print(f"Final shape: {df.shape}")

print("\n🎉 All CSV files processed successfully!")
