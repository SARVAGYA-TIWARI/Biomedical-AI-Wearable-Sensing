import os
import glob
import pandas as pd

def check_alignment():
    ecg_root = r"d:\BTP\diabetes_subset_ecg_data"
    print("Checking ECG session dates vs glucose dates...")
    for subj_num in sorted(os.listdir(ecg_root)):
        s_dir = os.path.join(ecg_root, subj_num, "sensor_data")
        if not os.path.isdir(s_dir):
            continue
        sessions = sorted(os.listdir(s_dir))
        subj_name = f"subject_{int(subj_num):02d}"
        print(f"\n{subj_name} (Folder {subj_num}): {len(sessions)} sessions")
        for sess in sessions:
            csv_files = os.listdir(os.path.join(s_dir, sess))
            print(f"   Session: {sess} -> {csv_files}")

if __name__ == '__main__':
    check_alignment()
