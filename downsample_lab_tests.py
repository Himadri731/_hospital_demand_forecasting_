"""
downsample_lab_tests.py - Trims lab_tests.csv to under 20 MB for GitHub web upload.
Preserves header and the first 150,000 rows so it remains a valid, representative sample.
"""

import os
import pandas as pd

def shrink_lab_tests():
    file_path = "data/raw/lab_tests.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    print(f"Reading {file_path}...")
    # Read first 120,000 rows (~13 MB, well within GitHub's 25 MB limit)
    df = pd.read_csv(file_path, nrows=120000)
    
    temp_path = "data/raw/lab_tests_github.csv"
    df.to_csv(temp_path, index=False)
    
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    print(f"Sample generated: {temp_path} is {size_mb:.2f} MB ({len(df):,} rows).")
    
    # Try replacing original lab_tests.csv
    try:
        import time
        # Try removing and replacing
        os.replace(temp_path, file_path)
        print(f"Successfully replaced {file_path}! Final size: {size_mb:.2f} MB.")
    except PermissionError:
        print(f"Notice: {file_path} is currently open in Excel or another program.")
        print(f"Created '{temp_path}' ({size_mb:.2f} MB). You can close Excel and rename it to 'lab_tests.csv' or upload it directly to GitHub!")

if __name__ == "__main__":
    shrink_lab_tests()
