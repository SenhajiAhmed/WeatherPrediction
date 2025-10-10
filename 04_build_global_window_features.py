#!/usr/bin/env python3
"""
04_build_global_window_features_optimized.py

Efficiently builds 7-day rolling-window features across large datasets (>1M rows)
by streaming grouped computations and writing incrementally to disk.

Run:
  python 04_build_global_window_features_optimized.py
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

INPUT_FILE = "era5_all_daily_features.csv"
OUTPUT_FILE = "era5_all_window_features.csv"
WINDOW_SIZE = 7
CHUNK_SAVE_SIZE = 10000  # write to disk every N samples

def process_group(lat, lon, group, window_size, feature_cols, target_cols):
    group = group.sort_values('date').reset_index(drop=True)
    if len(group) <= window_size:
        return pd.DataFrame()

    rows = []
    for i in range(window_size, len(group)):
        window = group.iloc[i - window_size:i]
        row = {
            'date': group['date'].iloc[i],
            'latitude': lat,
            'longitude': lon,
        }
        for col in feature_cols:
            vals = window[col].values
            row[f"{col}_mean"] = np.mean(vals)
            row[f"{col}_std"] = np.std(vals)
            row[f"{col}_min"] = np.min(vals)
            row[f"{col}_max"] = np.max(vals)
            row[f"{col}_first"] = vals[0]
            row[f"{col}_last"] = vals[-1]
            row[f"{col}_trend"] = vals[-1] - vals[0]
            if len(vals) >= 3:
                row[f"{col}_recent_3d"] = np.mean(vals[-3:])
        # Targets
        target_vals = group[target_cols].iloc[i].values
        for j, tcol in enumerate(target_cols):
            row[tcol] = target_vals[j]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run 03_concat_daily_features.py first.")

    df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
    print(f"Loaded {len(df):,} rows from {INPUT_FILE}")

    # Identify features and targets
    target_cols = [c for c in df.columns if c.endswith("_next")]
    feature_cols = [c for c in df.columns if c not in ['date', 'latitude', 'longitude'] + target_cols]
    print(f"Feature cols: {len(feature_cols)}, Target cols: {len(target_cols)}")

    # Prepare output
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Process by (latitude, longitude) group
    buffer = []
    total_rows = 0
    for (lat, lon), group in tqdm(df.groupby(['latitude', 'longitude'], sort=False), desc="Processing locations"):
        out_df = process_group(lat, lon, group, WINDOW_SIZE, feature_cols, target_cols)
        if out_df.empty:
            continue
        buffer.append(out_df)

        # Write periodically to avoid memory build-up
        if sum(len(b) for b in buffer) >= CHUNK_SAVE_SIZE:
            merged = pd.concat(buffer, ignore_index=True)
            merged = merged.astype({col: 'float32' for col in merged.select_dtypes('float64').columns})
            merged.to_csv(OUTPUT_FILE, mode='a', index=False, header=not os.path.exists(OUTPUT_FILE))
            total_rows += len(merged)
            buffer = []
            print(f"💾 Saved {total_rows:,} rows so far...")

    # Final flush
    if buffer:
        merged = pd.concat(buffer, ignore_index=True)
        merged = merged.astype({col: 'float32' for col in merged.select_dtypes('float64').columns})
        merged.to_csv(OUTPUT_FILE, mode='a', index=False, header=not os.path.exists(OUTPUT_FILE))
        total_rows += len(merged)
        print(f"💾 Final save. Total: {total_rows:,} rows")

    print(f"✅ All done! Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
