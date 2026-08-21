import os
import glob
import pandas as pd

def search_files():
    print("Searching for CSV and data files in d:\\BTP...")
    for root, dirs, files in os.walk(r"d:\BTP"):
        for f in files:
            if f.endswith(('.csv', '.parquet', '.tsv', '.txt', '.json', '.zip', '.gz')):
                p = os.path.join(root, f)
                size_mb = os.path.getsize(p) / (1024*1024)
                print(f"  {os.path.relpath(p, r'd:\BTP')} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    search_files()
