import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nhanes')

def compute_circadian_metrics(sub_df, start_time_str):
    """
    Computes parametric (Cosinor) and non-parametric (IS, IV, RA, M10, L5)
    circadian rhythm features for a single participant across their 7-day hourly record.
    """
    n_points = len(sub_df)
    if n_points < 48: # At least 2 full days of hourly data required
        return None

    # Determine hour-of-day for each row
    try:
        parts = start_time_str.split(':')
        h_start = int(parts[0])
    except Exception:
        h_start = 12 # fallback noon
    
    hours_of_day = np.array([(h_start + i) % 24 for i in range(n_points)], dtype=float)
    activity = sub_df['PAXMTSH'].values.astype(float) # MIMS triaxial acceleration
    wake_min = sub_df['PAXWWMH'].values.astype(float)
    sleep_min = sub_df['PAXSWMH'].values.astype(float)

    # Clean missing / negative values
    activity = np.nan_to_num(activity, nan=0.0)
    activity = np.maximum(activity, 0.0)
    wake_min = np.nan_to_num(wake_min, nan=0.0)
    sleep_min = np.nan_to_num(sleep_min, nan=0.0)

    mean_act = np.mean(activity)
    var_act = np.var(activity, ddof=0)
    if var_act < 1e-6:
        return None

    # 1. Non-Parametric Circadian Rhythm Analysis (NPCRA)
    # ── Interdaily Stability (IS) ──────────────────────────────────────────
    # Hourly 24-h profile
    hourly_means = np.zeros(24)
    for h in range(24):
        mask = (hours_of_day == h)
        if np.any(mask):
            hourly_means[h] = np.mean(activity[mask])
        else:
            hourly_means[h] = mean_act

    is_num = n_points * np.sum((hourly_means - mean_act) ** 2)
    is_den = 24.0 * np.sum((activity - mean_act) ** 2)
    interdaily_stability = float(is_num / is_den) if is_den > 0 else np.nan

    # ── Intradaily Variability (IV) ────────────────────────────────────────
    # Hour-to-hour fragmentation
    iv_num = n_points * np.sum((activity[1:] - activity[:-1]) ** 2)
    iv_den = (n_points - 1) * np.sum((activity - mean_act) ** 2)
    intradaily_variability = float(iv_num / iv_den) if iv_den > 0 else np.nan

    # ── M10 (Most active 10 hours) and L5 (Least active 5 hours) ──────────
    # Calculate 10-hour and 5-hour rolling averages on the 24-hour profile (circular)
    profile_extended = np.tile(hourly_means, 2) # double 24h to handle wrap-around
    
    # M10
    m10_vals = [np.mean(profile_extended[i:i+10]) for i in range(24)]
    m10_onset = int(np.argmax(m10_vals))
    m10_value = float(np.max(m10_vals))

    # L5
    l5_vals = [np.mean(profile_extended[i:i+5]) for i in range(24)]
    l5_onset = int(np.argmin(l5_vals))
    l5_value = float(np.min(l5_vals))

    # Relative Amplitude (RA)
    denom = (m10_value + l5_value)
    relative_amplitude = float((m10_value - l5_value) / denom) if denom > 0 else np.nan

    # 2. Parametric Cosinor Model
    # Fit y(t) = M + beta*cos(2pi*t/24) + gamma*sin(2pi*t/24) using linear regression
    # t is hour of day
    cos_t = np.cos(2 * np.pi * hours_of_day / 24.0)
    sin_t = np.sin(2 * np.pi * hours_of_day / 24.0)
    X = np.column_stack([np.ones(n_points), cos_t, sin_t])
    try:
        # Solve OLS: (X'X)^-1 X'y
        beta_hat, residuals, rank, s = np.linalg.lstsq(X, activity, rcond=None)
        mesor = float(beta_hat[0])
        b_cos = float(beta_hat[1])
        b_sin = float(beta_hat[2])
        amplitude = float(np.sqrt(b_cos**2 + b_sin**2))
        
        # Acrophase: peak time in hours [0, 24)
        phi_rad = np.arctan2(b_sin, b_cos) # in radians [-pi, pi]
        if phi_rad < 0:
            phi_rad += 2 * np.pi
        acrophase_hours = float((phi_rad * 24.0) / (2 * np.pi))

        # Goodness of fit (R-squared of 24-h cosinor)
        y_pred = X @ beta_hat
        ss_res = np.sum((activity - y_pred) ** 2)
        ss_tot = np.sum((activity - mean_act) ** 2)
        cosinor_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    except Exception:
        mesor = mean_act
        amplitude = np.nan
        acrophase_hours = np.nan
        cosinor_r2 = np.nan

    # 3. Sedentary Dynamics & Sleep Metrics
    # Define sedentary threshold: lower 25th percentile of waking population MIMS (~100)
    is_waking = (wake_min >= 30.0) # at least 30 min awake in that hour
    sedentary_hours = is_waking & (activity < 100.0)
    sedentary_pct = float(np.sum(sedentary_hours) / max(np.sum(is_waking), 1) * 100.0)

    # Sedentary bout duration (longest continuous sequence of sedentary hours)
    max_sed_bout = 0
    curr_sed_bout = 0
    for s_flag in sedentary_hours:
        if s_flag:
            curr_sed_bout += 1
            if curr_sed_bout > max_sed_bout:
                max_sed_bout = curr_sed_bout
        else:
            curr_sed_bout = 0

    # Total sleep hours / day
    total_sleep_hours = float(np.sum(sleep_min) / 60.0)
    valid_days = n_points / 24.0
    mean_nightly_sleep_hours = float(total_sleep_hours / valid_days) if valid_days > 0 else np.nan

    # Nighttime awakenings (wake wear during sleep-dominant hours 23:00 to 06:00)
    night_mask = (hours_of_day >= 23) | (hours_of_day < 6)
    night_wake_min_per_night = float(np.sum(wake_min[night_mask]) / valid_days) if valid_days > 0 else np.nan

    return {
        'circadian_mesor': mesor,
        'circadian_amplitude': amplitude,
        'circadian_acrophase': acrophase_hours,
        'circadian_r2': cosinor_r2,
        'interdaily_stability_IS': interdaily_stability,
        'intradaily_variability_IV': intradaily_variability,
        'relative_amplitude_RA': relative_amplitude,
        'm10_value': m10_value,
        'm10_onset': m10_onset,
        'l5_value': l5_value,
        'l5_onset': l5_onset,
        'sedentary_pct': sedentary_pct,
        'max_sedentary_bout_hours': max_sed_bout,
        'mean_nightly_sleep_hours': mean_nightly_sleep_hours,
        'nocturnal_wake_min': night_wake_min_per_night,
        'hourly_data_points': n_points
    }

def process_cycle_paxhr(cycle_suffix, target_seqns):
    print(f"\n=======================================================")
    print(f"Processing PAXHR for Cycle {cycle_suffix} ...")
    print(f"=======================================================")
    
    # 1. Load Header to get PAXFTIME (start time)
    paxhd_path = os.path.join(DATA_DIR, f'PAXHD_{cycle_suffix}.XPT')
    df_hd = pd.read_sas(paxhd_path, encoding='iso-8859-1')
    start_times = {}
    for _, row in df_hd.iterrows():
        s = row['SEQN']
        ft = row['PAXFTIME']
        if isinstance(ft, bytes):
            ft = ft.decode('utf-8')
        start_times[s] = str(ft).strip()

    # 2. Stream PAXHR in chunks
    paxhr_path = os.path.join(DATA_DIR, f'PAXHR_{cycle_suffix}.XPT')
    print(f"Streaming {paxhr_path} (size: {os.path.getsize(paxhr_path)/(1024*1024):.1f} MB) ...")
    
    chunk_iter = pd.read_sas(paxhr_path, format='xport', iterator=True)
    chunk_size = 50000
    
    records_by_seqn = {}
    total_rows = 0
    t0 = time.time()
    
    while True:
        try:
            chunk = chunk_iter.get_chunk(chunk_size)
        except StopIteration:
            break
            
        total_rows += len(chunk)
        # Filter only to target SEQNs present in our cohort
        filtered = chunk[chunk['SEQN'].isin(target_seqns)]
        if len(filtered) > 0:
            for seqn, group in filtered.groupby('SEQN'):
                if seqn not in records_by_seqn:
                    records_by_seqn[seqn] = []
                records_by_seqn[seqn].append(group[['PAXMTSH', 'PAXWWMH', 'PAXSWMH', 'PAXNWMH']])
                
        if total_rows % 200000 == 0:
            print(f"  Processed {total_rows:,} rows | Extracted {len(records_by_seqn):,} target participants ...", flush=True)

    print(f"Total rows scanned: {total_rows:,} in {time.time()-t0:.1f}s")
    print(f"Found {len(records_by_seqn):,} target participants with hourly records. Calculating features ...")

    feature_list = []
    for seqn, list_of_chunks in records_by_seqn.items():
        sub_df = pd.concat(list_of_chunks, ignore_index=True)
        stime = start_times.get(seqn, "12:00:00")
        feats = compute_circadian_metrics(sub_df, stime)
        if feats is not None:
            feats['SEQN'] = seqn
            feature_list.append(feats)

    features_df = pd.DataFrame(feature_list)
    print(f"Cycle {cycle_suffix} complete: {len(features_df)} participants with full circadian features.")
    return features_df

def main():
    # Load Master Dataset to get Target SEQNs
    master_path = os.path.join(DATA_DIR, 'nhanes_unified_master.parquet')
    if not os.path.exists(master_path):
        master_path = os.path.join(DATA_DIR, 'nhanes_unified_master.csv')
        master = pd.read_csv(master_path)
    else:
        master = pd.read_parquet(master_path)

    print(f"Master dataset loaded: {len(master)} participants.")
    target_seqns = set(master['SEQN'].values)
    print(f"Target SEQNs to extract: {len(target_seqns)}")

    df_g = process_cycle_paxhr('G', target_seqns)
    df_h = process_cycle_paxhr('H', target_seqns)

    circadian_df = pd.concat([df_g, df_h], ignore_index=True)
    print(f"\nTotal Circadian Features Extracted: {len(circadian_df):,} participants")

    # Save features standalone
    out_circ_path = os.path.join(DATA_DIR, 'nhanes_circadian_features.parquet')
    circadian_df.to_parquet(out_circ_path, index=False)
    print(f"Saved circadian features to: {out_circ_path}")

    # Merge into master dataset
    master_enriched = master.merge(circadian_df, on='SEQN', how='left')
    enriched_csv = os.path.join(DATA_DIR, 'nhanes_unified_master_with_circadian.csv')
    enriched_parquet = os.path.join(DATA_DIR, 'nhanes_unified_master_with_circadian.parquet')
    master_enriched.to_csv(enriched_csv, index=False)
    master_enriched.to_parquet(enriched_parquet, index=False)
    print(f"Saved enriched master dataset to:")
    print(f"  • {enriched_csv} ({os.path.getsize(enriched_csv)/(1024*1024):.2f} MB)")
    print(f"  • {enriched_parquet} ({os.path.getsize(enriched_parquet)/(1024*1024):.2f} MB)")

    print("\n" + "=" * 70)
    print("CIRCADIAN & WEARABLE FEATURE SUMMARY ON ADULT COHORT (AGES 18-65)")
    print("=" * 70)
    cohort = master_enriched[master_enriched['is_target_adult']].copy()
    print(f"Adult Cohort with complete features: {cohort['circadian_mesor'].notna().sum():,} / {len(cohort):,}")
    
    metrics = [
        'circadian_mesor', 'circadian_amplitude', 'circadian_acrophase', 'circadian_r2',
        'interdaily_stability_IS', 'intradaily_variability_IV', 'relative_amplitude_RA',
        'm10_value', 'l5_value', 'sedentary_pct', 'max_sedentary_bout_hours', 'mean_nightly_sleep_hours'
    ]
    print(cohort[metrics].describe().T[['mean', 'std', 'min', '50%', 'max']])

if __name__ == '__main__':
    main()
