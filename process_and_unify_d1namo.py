import os
import glob
import zipfile
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_hrv_from_ecg_file(ecg_path, chunk_size=500000, fs=250.0):
    """
    Process large ECG CSV file in chunks and extract 5-minute windowed HRV metrics.
    """
    all_rr_rows = []
    
    try:
        # Read in chunks to keep memory usage low and processing fast
        for chunk in pd.read_csv(ecg_path, chunksize=chunk_size):
            if len(chunk) < fs * 10:
                continue
            
            ecg_arr = chunk["EcgWaveform"].values
            time_arr = chunk["Time"].values
            
            # Bandpass filter
            filtered = bandpass_filter(ecg_arr, lowcut=0.5, highcut=40.0, fs=fs)
            
            # Derivative + squaring
            diff_sig = np.diff(filtered)
            squared = diff_sig ** 2
            win_size = int(0.15 * fs)
            kernel = np.ones(win_size) / win_size
            integrated = np.convolve(squared, kernel, mode='same')
            
            # Peak detection
            min_dist = int(0.35 * fs)
            threshold = np.mean(integrated) + 0.3 * np.std(integrated)
            peaks, _ = find_peaks(integrated, distance=min_dist, height=threshold)
            
            if len(peaks) < 5:
                continue
            
            # Refine peak locations
            r_peaks = []
            for p in peaks:
                start = max(0, p - 15)
                end = min(len(filtered), p + 15)
                r_peak = start + np.argmax(filtered[start:end])
                r_peaks.append(r_peak)
            r_peaks = np.array(r_peaks)
            
            # Parse times
            peak_times = pd.to_datetime(time_arr[r_peaks], format="%d/%m/%Y %H:%M:%S.%f")
            rr_intervals = np.diff(peak_times.values).astype('timedelta64[ms]').astype(float)
            
            valid_mask = (rr_intervals >= 300) & (rr_intervals <= 1800)
            if np.sum(valid_mask) > 0:
                valid_rr = rr_intervals[valid_mask]
                valid_times = peak_times[1:][valid_mask]
                for t, r in zip(valid_times, valid_rr):
                    all_rr_rows.append((t, r))
    except Exception as e:
        print(f"    Error processing {ecg_path}: {e}")
        return pd.DataFrame()
    
    if not all_rr_rows:
        return pd.DataFrame()
    
    rr_df = pd.DataFrame(all_rr_rows, columns=["Time", "RR"])
    rr_df["Time_5min"] = rr_df["Time"].dt.floor("5min")
    
    metrics = []
    for t5, group in rr_df.groupby("Time_5min"):
        rr = group["RR"].values
        if len(rr) >= 10:
            hr = 60000.0 / np.mean(rr)
            sdnn = np.std(rr, ddof=1)
            diff_rr = np.diff(rr)
            rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else np.nan
            pnn50 = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100.0 if len(diff_rr) > 0 else np.nan
            metrics.append({
                "Time": t5,
                "ECG_HR": round(hr, 2),
                "ECG_SDNN": round(sdnn, 2),
                "ECG_RMSSD": round(rmssd, 2),
                "ECG_pNN50": round(pnn50, 2),
                "ECG_n_beats": len(rr)
            })
            
    return pd.DataFrame(metrics)

def load_summary_data(summary_zip_dir):
    """
    Extract and parse all Summary CSVs from downloads zip files.
    """
    print("Loading Zephyr Summary data...")
    summary_dfs = []
    for zpath in glob.glob(os.path.join(summary_zip_dir, "*Summary*.zip")):
        with zipfile.ZipFile(zpath, 'r') as z:
            for name in z.namelist():
                if name.endswith("Summary.csv"):
                    with z.open(name) as f:
                        df = pd.read_csv(f)
                        df["Time"] = pd.to_datetime(df["Time"], format="%d/%m/%Y %H:%M:%S")
                        summary_dfs.append(df)
                        print(f"  Loaded {name}: {len(df)} rows")
                        
    if not summary_dfs:
        return pd.DataFrame()
    
    all_summary = pd.concat(summary_dfs, ignore_index=True).sort_values("Time")
    
    # Clean sentinel values
    all_summary["HR"] = all_summary["HR"].replace(0, np.nan)
    all_summary["HRV"] = all_summary["HRV"].replace(65535, np.nan)
    all_summary["CoreTemp"] = all_summary["CoreTemp"].replace(6553.5, np.nan)
    all_summary["SkinTemp"] = all_summary["SkinTemp"].replace(-3276.8, np.nan)
    all_summary["GSR"] = all_summary["GSR"].replace(65535, np.nan)
    
    # Resample to 5-minute regular grid
    sensor_cols = ["HR", "BR", "DeviceTemp", "Activity", "Posture", "CoreTemp"]
    summary_5min = all_summary.set_index("Time")[sensor_cols].resample("5min").mean()
    # Gap fill short dropouts (<= 15 min = 3 steps)
    summary_5min = summary_5min.interpolate(method="linear", limit=3, limit_area="inside")
    return summary_5min.reset_index()

def build_complete_multimodal_dataset():
    out_dir = r"d:\BTP\data"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Glucose
    print("\n--- 1. Loading Glucose Data for All 9 Subjects ---")
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
    
    glucose_list = []
    for subj_id, filepath in glucose_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["subject_id"] = subj_id
            df["Time"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S")
            if df["glucose"].median() < 35:
                df["glucose_mgdl"] = df["glucose"] * 18.0182
            else:
                df["glucose_mgdl"] = df["glucose"]
            
            cgm = df[df["type"] == "cgm"].sort_values("Time").reset_index(drop=True)
            glucose_list.append(cgm[["subject_id", "Time", "glucose_mgdl"]])
            print(f"  {subj_id}: {len(cgm)} readings")
            
    master_glucose = pd.concat(glucose_list, ignore_index=True)
    
    # 2. Extract ECG HRV metrics across all subjects
    print("\n--- 2. Processing Raw ECG Biosignals (250 Hz) ---")
    ecg_root = r"d:\BTP\diabetes_subset_ecg_data"
    ecg_hrv_list = []
    
    for subj_folder in sorted(os.listdir(ecg_root)):
        subj_num = int(subj_folder)
        subj_id = f"subject_{subj_num:02d}"
        s_dir = os.path.join(ecg_root, subj_folder, "sensor_data")
        if not os.path.isdir(s_dir):
            continue
        
        sessions = sorted(os.listdir(s_dir))
        print(f"  Processing {subj_id} ({len(sessions)} sessions)...")
        t_sub0 = time.time()
        for sess in sessions:
            csv_path = os.path.join(s_dir, sess, f"{sess}_ECG.csv")
            if os.path.exists(csv_path):
                hrv_df = extract_hrv_from_ecg_file(csv_path)
                if len(hrv_df) > 0:
                    hrv_df["subject_id"] = subj_id
                    ecg_hrv_list.append(hrv_df)
        print(f"    Finished {subj_id} in {time.time() - t_sub0:.1f}s")
        
    master_ecg = pd.concat(ecg_hrv_list, ignore_index=True) if ecg_hrv_list else pd.DataFrame()
    print(f"Total 5-min ECG HRV windows extracted: {len(master_ecg)}")
    
    # 3. Load Summary Data
    print("\n--- 3. Loading Zephyr Summary Sensors ---")
    summary_5min = load_summary_data(r"C:\Users\HP\Downloads")
    
    # 4. Synchronize onto a 5-minute regular grid
    print("\n--- 4. Building Regular 5-Minute Synchronized Multimodal Grid ---")
    all_subject_grids = []
    
    for subj_id in master_glucose["subject_id"].unique():
        g_sub = master_glucose[master_glucose["subject_id"] == subj_id].set_index("Time").sort_index()
        # Resample glucose onto regular 5-minute grid
        g_reg = g_sub[["glucose_mgdl"]].resample("5min").mean()
        # Gap fill short gaps <= 15 min (3 steps)
        g_reg["glucose_mgdl"] = g_reg["glucose_mgdl"].interpolate(method="linear", limit=3, limit_area="inside")
        g_reg["subject_id"] = subj_id
        
        # Merge ECG HRV metrics
        if len(master_ecg) > 0 and subj_id in master_ecg["subject_id"].values:
            ecg_sub = master_ecg[master_ecg["subject_id"] == subj_id].set_index("Time").sort_index()
            ecg_sub = ecg_sub.drop(columns=["subject_id"])
            g_reg = g_reg.join(ecg_sub, how="left")
        else:
            for col in ["ECG_HR", "ECG_SDNN", "ECG_RMSSD", "ECG_pNN50", "ECG_n_beats"]:
                g_reg[col] = np.nan
                
        # Merge Summary sensors if available
        if len(summary_5min) > 0:
            sum_sub = summary_5min.set_index("Time").sort_index()
            g_reg = g_reg.join(sum_sub, how="left")
        else:
            for col in ["HR", "BR", "DeviceTemp", "Activity", "Posture", "CoreTemp"]:
                g_reg[col] = np.nan
                
        # Unified Heart Rate: use Summary HR if available, else ECG_HR
        if "HR" in g_reg.columns and "ECG_HR" in g_reg.columns:
            g_reg["HR_unified"] = g_reg["HR"].combine_first(g_reg["ECG_HR"])
        elif "ECG_HR" in g_reg.columns:
            g_reg["HR_unified"] = g_reg["ECG_HR"]
        else:
            g_reg["HR_unified"] = g_reg.get("HR", np.nan)
            
        all_subject_grids.append(g_reg.reset_index())
        
    master_df = pd.concat(all_subject_grids, ignore_index=True)
    master_df = master_df.sort_values(["subject_id", "Time"]).reset_index(drop=True)
    
    # 5. Feature Engineering
    print("\n--- 5. Engineering Temporal, Circadian & Multimodal Features ---")
    # Time of day and circadian harmonics (SweetDeep approach)
    hour = master_df["Time"].dt.hour + master_df["Time"].dt.minute / 60.0
    phi = 2 * np.pi * hour / 24.0
    master_df["hour_sin"] = np.sin(phi)
    master_df["hour_cos"] = np.cos(phi)
    master_df["sin_2phi"] = np.sin(2 * phi)
    master_df["cos_2phi"] = np.cos(2 * phi)
    master_df["is_night"] = ((hour >= 23) | (hour <= 6)).astype(int)
    master_df["day_of_week"] = master_df["Time"].dt.dayofweek
    
    # Glucose history lags & rolling stats (Phase 1)
    g_grp = master_df.groupby("subject_id")["glucose_mgdl"]
    for lag_min in [5, 10, 15, 20, 25, 30]:
        step = lag_min // 5
        master_df[f"glucose_lag_{lag_min}min"] = g_grp.shift(step)
        
    master_df["glucose_roll_mean_30min"] = g_grp.shift(1).rolling(6, min_periods=3).mean()
    master_df["glucose_roll_std_30min"] = g_grp.shift(1).rolling(6, min_periods=3).std()
    master_df["glucose_roc_15min"] = (g_grp.shift(1) - g_grp.shift(3)) / 15.0  # mg/dL/min
    
    # Sensor rolling stats
    for col in ["HR_unified", "ECG_SDNN", "ECG_RMSSD", "DeviceTemp", "Activity", "BR"]:
        if col in master_df.columns:
            s_grp = master_df.groupby("subject_id")[col]
            master_df[f"{col}_roll15"] = s_grp.shift(1).rolling(3, min_periods=1).mean()
            master_df[f"{col}_roll30"] = s_grp.shift(1).rolling(6, min_periods=1).mean()
            
    # Targets for Phase 1: 30-min (6 steps) and 60-min (12 steps) ahead forecasting
    master_df["target_30min"] = g_grp.shift(-6)
    master_df["target_60min"] = g_grp.shift(-12)
    
    # Targets for Phase 2:
    # 1. Glucose Range (CGM Time-in-Range consensus)
    # Low (<70), Target (70-180), High (>180)
    def assign_cgm_range(val):
        if pd.isna(val):
            return np.nan
        if val < 70:
            return 0  # Hypoglycemic
        elif val <= 180:
            return 1  # Target / Normal
        else:
            return 2  # Hyperglycemic
            
    master_df["range_cgm"] = master_df["glucose_mgdl"].apply(assign_cgm_range)
    
    # 2. Glucose Trend (30-min and 60-min ahead rate of change)
    # Clinically grounded threshold: +/- 1.0 mg/dL/min (+/- 30 mg/dL over 30 min)
    delta_30 = master_df["target_30min"] - master_df["glucose_mgdl"]
    rate_30 = delta_30 / 30.0
    
    def assign_trend(rate):
        if pd.isna(rate):
            return np.nan
        if rate > 1.0:
            return 2  # Increasing
        elif rate < -1.0:
            return 0  # Decreasing
        else:
            return 1  # Stable
            
    master_df["trend_30min"] = rate_30.apply(assign_trend)
    
    delta_60 = master_df["target_60min"] - master_df["glucose_mgdl"]
    rate_60 = delta_60 / 60.0
    master_df["trend_60min"] = rate_60.apply(assign_trend)
    
    # Save master dataset
    parquet_path = os.path.join(out_dir, "d1namo_multimodal_master.parquet")
    csv_path = os.path.join(out_dir, "d1namo_multimodal_master.csv")
    master_df.to_parquet(parquet_path, index=False)
    master_df.to_csv(csv_path, index=False)
    
    print(f"\n==========================================")
    print(f"MULTIMODAL MASTER DATASET CREATED SUCCESSFULLY!")
    print(f"Total Rows: {len(master_df)}")
    print(f"Subjects: {master_df['subject_id'].value_counts().to_dict()}")
    print(f"Columns ({len(master_df.columns)}): {list(master_df.columns)}")
    print(f"Saved to: {parquet_path}")
    print(f"Saved to: {csv_path}")
    print(f"==========================================")
    return master_df

if __name__ == '__main__':
    t_start = time.time()
    df = build_complete_multimodal_dataset()
    print(f"\nTotal pipeline elapsed time: {time.time() - t_start:.1f}s")
