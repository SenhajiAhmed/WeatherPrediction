#!/usr/bin/env python3
"""
Script to concatenate ERA5 CSV files in chronological order.
Combines all CSVs matching the pattern 'era5_full_dataset_YYYY_MM_window_features.csv'
from the 'era5_features' directory, with older data at the top and newer data at the bottom.
"""

import pandas as pd
import glob
import os
import re
from pathlib import Path
from tqdm import tqdm

def extract_date_from_filename(filename):
    """
    Extract date information from filename for sorting.
    Pattern: era5_full_dataset_YYYY_MM_window_features.csv
    """
    # Extract YYYY_MM pattern from the filename
    pattern = r'era5_full_dataset_(\d{4})_(\d{2})_window_features'
    match = re.search(pattern, filename)
    
    if match:
        year, month = match.groups()
        return f"{year}{month}"
    
    # If no date found, return filename for alphabetical sorting
    return filename

def concatenate_csvs(input_dir='era5_features', 
                     pattern='era5_full_dataset_*_*_window_features.csv',
                     output_file='era5_concatenated_output.csv'):
    """
    Concatenate all matching CSV files in chronological order.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the CSV files
    pattern : str
        Pattern to match CSV filenames
    output_file : str
        Output filename for concatenated CSV
    """
    
    # Create full path pattern
    search_pattern = os.path.join(input_dir, pattern)
    
    # Find all matching files
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No files found matching pattern: {search_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    
    # Sort files chronologically based on filename
    csv_files.sort(key=lambda x: extract_date_from_filename(os.path.basename(x)))
    
    # Display files in order
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Read and concatenate all CSVs
    print("\nConcatenating files...")
    dataframes = []
    
    for file in tqdm(csv_files, desc="Loading CSV files", unit="file"):
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            tqdm.write(f"  ✓ Loaded {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            tqdm.write(f"  ✗ Error reading {os.path.basename(file)}: {e}")
    
    if not dataframes:
        print("No data to concatenate!")
        return
    
    # Concatenate all dataframes
    print("\nConcatenating dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"  ✓ Concatenation complete ({len(combined_df)} total rows)")
    
    # Save to output file
    print(f"\nSaving to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully concatenated {len(dataframes)} files")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Output saved to: {output_file}")

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = 'era5_features'
    FILE_PATTERN = 'era5_full_dataset_*_*_window_features.csv'
    OUTPUT_FILE = 'era5_concatenated_output.csv'
    
    # Create input directory if it doesn't exist (for testing)
    Path(INPUT_DIR).mkdir(exist_ok=True)
    
    # Run concatenation
    concatenate_csvs(
        input_dir=INPUT_DIR,
        pattern=FILE_PATTERN,
        output_file=OUTPUT_FILE
    )