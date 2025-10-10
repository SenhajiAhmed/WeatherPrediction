#!/usr/bin/env python3
"""
03_concat_daily_features.py

Concatenates all *_daily_features.csv files from 'era5_features/'
into a single continuous dataset sorted by date and location.

Run:
  python 03_concat_daily_features.py
"""

import os
import re
import glob
import pandas as pd
from tqdm import tqdm

INPUT_DIR = "era5_features"
OUTPUT_FILE = "era5_all_daily_features.csv"

def extract_date_from_filename(filename):
    match = re.search(r'era5_full_dataset_(\d{4})_(\d{2})', filename)
    return int(match.group(1) + match.group(2)) if match else 0

def concatenate_daily_features():
    pattern = os.path.join(INPUT_DIR, "*_daily_features.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        print(f"No daily features found in {INPUT_DIR}")
        return
    
    csv_files.sort(key=lambda f: extract_date_from_filename(os.path.basename(f)))

    print(f"Found {len(csv_files)} monthly daily features files:")
    for f in csv_files:
        print("  •", os.path.basename(f))
    
    dfs = []
    for f in tqdm(csv_files, desc="Loading files", unit="file"):
        df = pd.read_csv(f, parse_dates=['date'])
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(['latitude', 'longitude', 'date'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"✅ Combined dataset shape: {combined.shape}")

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved concatenated file → {OUTPUT_FILE}")

if __name__ == "__main__":
    concatenate_daily_features()
