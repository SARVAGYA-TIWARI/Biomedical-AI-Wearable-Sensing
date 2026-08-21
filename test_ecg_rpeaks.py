import os
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

def process_ecg_chunk(time_arr, ecg_arr, fs=250.0):
    if len(ecg_arr) < fs * 10:
        return pd.DataFrame()
    
    # 1. Bandpass filter
    filtered = bandpass_filter(ecg_arr, lowcut=0.5, highcut=40.0, fs=fs)
    
    # 2. Derivative + squaring
    diff_sig = np.diff(filtered)
    squared = diff_sig ** 2
    win_size = int(0.15 * fs)
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode='same')
    
    # 3. Peak detection
    min_dist = int(0.35 * fs)
    threshold = np.mean(integrated) + 0.3 * np.std(integrated)
    peaks, _ = find_peaks(integrated, distance=min_dist, height=threshold)
    
    if len(peaks) < 5:
        return pd.DataFrame()
    
    # Refine peak locations on filtered signal
    r_peaks = []
    for p in peaks:
        start = max(0, p - 15)
        end = min(len(filtered), p + 15)
        r_peak = start + np.argmax(filtered[start:end])
        r_peaks.append(r_peak)
    r_peaks = np.array(r_peaks)
    
    peak_times = pd.to_datetime(time_arr[r_peaks])
    rr_intervals = np.diff(peak_times.values).astype('timedelta64[ms]').astype(float)
    
    valid_mask = (rr_intervals >= 300) & (rr_intervals <= 1800)
    if np.sum(valid_mask) < 5:
        return pd.DataFrame()
    
    valid_rr = rr_intervals[valid_mask]
    valid_times = peak_times[1:][valid_mask]
    
    rr_df = pd.DataFrame({"Time": valid_times, "RR": valid_rr})
    rr_df["Time_5min"] = rr_df["Time"].dt.floor("5min")
    
    rows = []
    for t5, group in rr_df.groupby("Time_5min"):
        rr = group["RR"].values
        if len(rr) >= 10:
            hr = 60000.0 / np.mean(rr)
            sdnn = np.std(rr, ddof=1)
            diff_rr = np.diff(rr)
            rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else np.nan
            pnn50 = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100.0 if len(diff_rr) > 0 else np.nan
            rows.append({
                "Time": t5,
                "HR": round(hr, 2),
                "SDNN": round(sdnn, 2),
                "RMSSD": round(rmssd, 2),
                "pNN50": round(pnn50, 2),
                "n_beats": len(rr)
            })
    
    return pd.DataFrame(rows)

if __name__ == '__main__':
    t0 = time.time()
    sample_file = r"d:\BTP\diabetes_subset_ecg_data\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_ECG.csv"
    print(f"Reading sample ECG: {sample_file} (500k rows)...")
    df = pd.read_csv(sample_file, nrows=500000)
    print(f"Read in {time.time() - t0:.2f}s. Processing R-peaks & HRV...")
    t1 = time.time()
    hrv_res = process_ecg_chunk(df["Time"].values, df["EcgWaveform"].values)
    print(f"HRV extracted in {time.time() - t1:.2f}s:")
    print(hrv_res)
