"""
replace_lab_tests.py - Replaces lab_tests.csv with the <25MB version (17.7 MB)
and cleans up any temporary files, keeping the project structure identical.
"""

import os
import shutil
import time

def main():
    target = "data/raw/lab_tests.csv"
    source = "data/raw/lab_tests_github.csv"
    
    if not os.path.exists(source):
        print(f"Source file {source} does not exist.")
        return

    # Try to overwrite target with source
    success = False
    for attempt in range(5):
        try:
            # Overwrite target with source
            shutil.copyfile(source, target)
            success = True
            break
        except PermissionError:
            time.sleep(1)
            
    if success:
        # Remove source to keep only the clean lab_tests.csv
        try:
            os.remove(source)
        except Exception:
            pass
        size_mb = os.path.getsize(target) / (1024 * 1024)
        print(f"Success! {target} is now {size_mb:.2f} MB (under 25 MB) and ready for GitHub.")
    else:
        print(f"Could not overwrite {target}. It may be currently open in another program.")

if __name__ == "__main__":
    main()
