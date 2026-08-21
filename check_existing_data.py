import os
import glob
import pandas as pd

print("=== Checking glucose files in Downloads ===")
for i in range(1, 9):
    p = rf"C:\Users\HP\Downloads\glucose ({i}).csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"glucose ({i}).csv: {len(df)} rows, cols={list(df.columns)}")
p0 = r"C:\Users\HP\Downloads\glucose.csv"
if os.path.exists(p0):
    df = pd.read_csv(p0)
    print(f"glucose.csv: {len(df)} rows, cols={list(df.columns)}")

print("\n=== Checking summary files in Downloads ===")
for p in glob.glob(r"C:\Users\HP\Downloads\*Summary*.zip"):
    print(f"Summary zip: {p}")

print("\n=== Checking ECG files in diabetes_subset_ecg_data ===")
for subj in sorted(os.listdir(r"d:\BTP\diabetes_subset_ecg_data")):
    sdir = os.path.join(r"d:\BTP\diabetes_subset_ecg_data", subj, "sensor_data")
    if os.path.exists(sdir):
        sessions = os.listdir(sdir)
        total_size = sum(os.path.getsize(os.path.join(sdir, sess, f)) for sess in sessions for f in os.listdir(os.path.join(sdir, sess)))
        print(f"Subject {subj}: {len(sessions)} sessions, {total_size/(1024*1024):.1f} MB")
