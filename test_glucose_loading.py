import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

def build_glucose_master():
    print("Loading and standardizing glucose data for all 9 subjects...")
    # Map downloads glucose files to subject IDs
    # Subject 001: glucose.csv
    # Subject 002: glucose (1).csv ... Subject 009: glucose (8).csv
    glucose_files = {
        "subject_01": r"C:\Users\HP\Downloads\glucose.csv",
        "subject_02": r"C:\Users\HP\Downloads\glucose (1).csv",
        "subject_03": r"C:\Users\HP\Downloads\glucose (2).csv",
        "subject_04": r"C:\Users\HP\Downloads\glucose (3).csv",
        "subject_05": r"C:\Users\HP\Downloads\glucose (4).csv",
        "subject_06": r"C:\Users\HP\Downloads\glucose (5).csv",
        "subject_07": r"C:\Users\HP\Downloads\glucose (6).csv",
        "subject_08": r"C:\Users\HP\Downloads\glucose (7).csv",
        "subject_09": r"C:\Users\HP\Downloads\glucose (8).csv",
    }
    
    dfs = []
    for subj_id, filepath in glucose_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["subject_id"] = subj_id
            df["Time"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S")
            # Convert mmol/L to mg/dL (clinical standard * 18.0182)
            # Check if values are in mmol/L (< 35) or already mg/dL
            if df["glucose"].median() < 35:
                df["glucose_mgdl"] = df["glucose"] * 18.0182
            else:
                df["glucose_mgdl"] = df["glucose"]
            
            # Keep CGM readings primarily
            df_cgm = df[df["type"] == "cgm"].copy()
            df_cgm = df_cgm.sort_values("Time").reset_index(drop=True)
            dfs.append(df_cgm)
            print(f"  {subj_id}: {len(df_cgm)} CGM rows, span: {df_cgm['Time'].min()} to {df_cgm['Time'].max()}, mean={df_cgm['glucose_mgdl'].mean():.1f} mg/dL")
    
    master_glucose = pd.concat(dfs, ignore_index=True).sort_values(["subject_id", "Time"])
    return master_glucose

if __name__ == '__main__':
    g = build_glucose_master()
    print(f"\nTotal CGM measurements across 9 subjects: {len(g)}")
