import os
import glob
import zipfile
import time
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_single_ecg_session(args):
    subj_id, ecg_path = args
    fs = 250.0
    chunk_size = 1000000
    all_rr_rows = []
    
    if not os.path.exists(ecg_path):
        return pd.DataFrame()
        
    try:
        for chunk in pd.read_csv(ecg_path, usecols=["Time", "EcgWaveform"], chunksize=chunk_size):
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
        print(f"Error in {ecg_path}: {e}")
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
                "ECG_n_beats": len(rr),
                "subject_id": subj_id
            })
            
    return pd.DataFrame(metrics)

def load_summary_data(summary_zip_dir):
    print("Loading Zephyr Summary data...")
    summary_dfs = []
    for zpath in glob.glob(os.path.join(summary_zip_dir, "*Summary*.zip")):
        with zipfile.ZipFile(zpath, 'r') as z:
            for name in z.namelist():
                if name.endswith("Summary.csv"):
                    with z.open(name) as f:
                        df = pd.read_csv(f)
                        df["Time"] = pd.to_datetime(df["Time"], format='mixed', dayfirst=True)
                        summary_dfs.append(df)
                        print(f"  Loaded {name}: {len(df)} rows")
                        
    if not summary_dfs:
        return pd.DataFrame()
    
    all_summary = pd.concat(summary_dfs, ignore_index=True).sort_values("Time")
    
    # Clean sentinels
    all_summary["HR"] = all_summary["HR"].replace(0, np.nan)
    all_summary["HRV"] = all_summary["HRV"].replace(65535, np.nan)
    all_summary["CoreTemp"] = all_summary["CoreTemp"].replace(6553.5, np.nan)
    all_summary["SkinTemp"] = all_summary["SkinTemp"].replace(-3276.8, np.nan)
    all_summary["GSR"] = all_summary["GSR"].replace(65535, np.nan)
    
    sensor_cols = ["HR", "BR", "DeviceTemp", "Activity", "Posture", "CoreTemp"]
    summary_5min = all_summary.set_index("Time")[sensor_cols].resample("5min").mean()
    summary_5min = summary_5min.interpolate(method="linear", limit=3, limit_area="inside")
    return summary_5min.reset_index()

def main():
    out_dir = r"d:\BTP\data"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    
    # 1. Glucose
    print("--- 1. Loading Glucose Data for All 9 Subjects ---")
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
    
    ecg_cache_path = os.path.join(out_dir, "d1namo_ecg_hrv_extracted.parquet")
    if os.path.exists(ecg_cache_path):
        print(f"\n--- 2. Loading Cached ECG HRV Metrics from {ecg_cache_path} ---")
        master_ecg = pd.read_parquet(ecg_cache_path)
        print(f"Loaded {len(master_ecg)} 5-min ECG windows from cache.")
    else:
        print("\n--- 2. Parallel ECG Biosignal Processing (16 cores, 8 workers) ---")
        ecg_root = r"d:\BTP\diabetes_subset_ecg_data"
        tasks = []
        for subj_folder in sorted(os.listdir(ecg_root)):
            subj_num = int(subj_folder)
            subj_id = f"subject_{subj_num:02d}"
            s_dir = os.path.join(ecg_root, subj_folder, "sensor_data")
            if not os.path.isdir(s_dir):
                continue
            sessions = sorted(os.listdir(s_dir))
            for sess in sessions:
                csv_path = os.path.join(s_dir, sess, f"{sess}_ECG.csv")
                if os.path.exists(csv_path):
                    tasks.append((subj_id, csv_path))
                    
        print(f"Total ECG session files to process: {len(tasks)}")
        ecg_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_single_ecg_session, task): task for task in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                completed += 1
                res = future.result()
                if len(res) > 0:
                    ecg_results.append(res)
                    print(f"  [{completed}/{len(tasks)}] Finished {task[0]} - {os.path.basename(task[1])}: {len(res)} 5-min windows")
                else:
                    print(f"  [{completed}/{len(tasks)}] Finished {task[0]} - {os.path.basename(task[1])}: (empty)")
                    
        master_ecg = pd.concat(ecg_results, ignore_index=True) if ecg_results else pd.DataFrame()
        if len(master_ecg) > 0:
            master_ecg.to_parquet(ecg_cache_path, index=False)
            print(f"Saved ECG cache to {ecg_cache_path}")
        print(f"\nTotal extracted ECG 5-min windows: {len(master_ecg)}")
    
    # 3. Summary Data
    print("\n--- 3. Loading Zephyr Summary Sensors ---")
    summary_5min = load_summary_data(r"C:\Users\HP\Downloads")
    
    # 4. Synchronize Multimodal Grid
    print("\n--- 4. Synchronizing Multimodal Grid per Subject ---")
    all_subject_grids = []
    
    for subj_id in master_glucose["subject_id"].unique():
        g_sub = master_glucose[master_glucose["subject_id"] == subj_id].set_index("Time").sort_index()
        g_reg = g_sub[["glucose_mgdl"]].resample("5min").mean()
        g_reg["glucose_mgdl"] = g_reg["glucose_mgdl"].interpolate(method="linear", limit=3, limit_area="inside")
        g_reg["subject_id"] = subj_id
        
        if len(master_ecg) > 0 and subj_id in master_ecg["subject_id"].values:
            ecg_sub = master_ecg[master_ecg["subject_id"] == subj_id].set_index("Time").sort_index()
            ecg_sub = ecg_sub.drop(columns=["subject_id"])
            g_reg = g_reg.join(ecg_sub, how="left")
        else:
            for col in ["ECG_HR", "ECG_SDNN", "ECG_RMSSD", "ECG_pNN50", "ECG_n_beats"]:
                g_reg[col] = np.nan
                
        if len(summary_5min) > 0:
            sum_sub = summary_5min.set_index("Time").sort_index()
            g_reg = g_reg.join(sum_sub, how="left")
        else:
            for col in ["HR", "BR", "DeviceTemp", "Activity", "Posture", "CoreTemp"]:
                g_reg[col] = np.nan
                
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
    print("\n--- 5. Engineering Features ---")
    hour = master_df["Time"].dt.hour + master_df["Time"].dt.minute / 60.0
    phi = 2 * np.pi * hour / 24.0
    master_df["hour_sin"] = np.sin(phi)
    master_df["hour_cos"] = np.cos(phi)
    master_df["sin_2phi"] = np.sin(2 * phi)
    master_df["cos_2phi"] = np.cos(2 * phi)
    master_df["is_night"] = ((hour >= 23) | (hour <= 6)).astype(int)
    master_df["day_of_week"] = master_df["Time"].dt.dayofweek
    
    g_grp = master_df.groupby("subject_id")["glucose_mgdl"]
    for lag_min in [5, 10, 15, 20, 25, 30]:
        step = lag_min // 5
        master_df[f"glucose_lag_{lag_min}min"] = g_grp.shift(step)
        
    master_df["glucose_roll_mean_30min"] = g_grp.shift(1).rolling(6, min_periods=3).mean()
    master_df["glucose_roll_std_30min"] = g_grp.shift(1).rolling(6, min_periods=3).std()
    master_df["glucose_roc_15min"] = (g_grp.shift(1) - g_grp.shift(3)) / 15.0
    
    for col in ["HR_unified", "ECG_SDNN", "ECG_RMSSD", "DeviceTemp", "Activity", "BR"]:
        if col in master_df.columns:
            s_grp = master_df.groupby("subject_id")[col]
            master_df[f"{col}_roll15"] = s_grp.shift(1).rolling(3, min_periods=1).mean()
            master_df[f"{col}_roll30"] = s_grp.shift(1).rolling(6, min_periods=1).mean()
            
    # Phase 1 targets (30 & 60 min ahead)
    master_df["target_30min"] = g_grp.shift(-6)
    master_df["target_60min"] = g_grp.shift(-12)
    
    # Phase 2 targets
    def assign_cgm_range(val):
        if pd.isna(val): return np.nan
        if val < 70: return 0
        elif val <= 180: return 1
        else: return 2
    master_df["range_cgm"] = master_df["glucose_mgdl"].apply(assign_cgm_range)
    
    delta_30 = master_df["target_30min"] - master_df["glucose_mgdl"]
    rate_30 = delta_30 / 30.0
    def assign_trend(rate):
        if pd.isna(rate): return np.nan
        if rate > 1.0: return 2
        elif rate < -1.0: return 0
        else: return 1
    master_df["trend_30min"] = rate_30.apply(assign_trend)
    
    delta_60 = master_df["target_60min"] - master_df["glucose_mgdl"]
    rate_60 = delta_60 / 60.0
    master_df["trend_60min"] = rate_60.apply(assign_trend)
    
    # Save
    parquet_path = os.path.join(out_dir, "d1namo_multimodal_master.parquet")
    csv_path = os.path.join(out_dir, "d1namo_multimodal_master.csv")
    master_df.to_parquet(parquet_path, index=False)
    master_df.to_csv(csv_path, index=False)
    
    print(f"\n==========================================")
    print(f"PARALLEL PREPROCESSING COMPLETED!")
    print(f"Total Rows: {len(master_df)}")
    print(f"Elapsed Time: {time.time() - t0:.1f}s")
    print(f"Saved: {parquet_path}")
    print(f"==========================================")

if __name__ == '__main__':
    main()
