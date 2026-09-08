import os
import sys
import time
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nhanes')

def build_sequences():
    print("=" * 70)
    print("EXTRACTING 168-HOUR WEARABLE ACTIGRAPHY SEQUENCES")
    print("=" * 70)
    
    master_path = os.path.join(DATA_DIR, 'nhanes_unified_master_with_circadian.parquet')
    master = pd.read_parquet(master_path)
    cohort = master[master['is_target_adult'] & master['circadian_mesor'].notna()].copy()
    cohort = cohort[cohort['metabolic_risk_class'].notna()].copy()
    
    target_seqns = set(cohort['SEQN'].values)
    print(f"Target participants in cohort: {len(target_seqns)}")

    # Map SEQN to order in cohort
    seqn_list = cohort['SEQN'].tolist()
    seqn_to_idx = {s: i for i, s in enumerate(seqn_list)}

    # Channels: 0: PAXMTSH (Activity MIMS), 1: PAXWWMH (Wake min), 2: PAXSWMH (Sleep min)
    # Shape: (N, 168, 3)
    N = len(cohort)
    T = 168
    C = 3
    sequence_array = np.zeros((N, T, C), dtype=np.float32)
    seqn_found = set()

    for cycle in ['G', 'H']:
        paxhr_path = os.path.join(DATA_DIR, f'PAXHR_{cycle}.XPT')
        print(f"\nStreaming {paxhr_path} ...")
        t0 = time.time()
        
        chunk_iter = pd.read_sas(paxhr_path, format='xport', iterator=True)
        chunk_size = 50000
        
        participant_buffers = {}
        
        while True:
            try:
                chunk = chunk_iter.get_chunk(chunk_size)
            except StopIteration:
                break
                
            filtered = chunk[chunk['SEQN'].isin(target_seqns)]
            if len(filtered) > 0:
                for seqn, grp in filtered.groupby('SEQN'):
                    if seqn not in participant_buffers:
                        participant_buffers[seqn] = []
                    # Keep values
                    arr = grp[['PAXMTSH', 'PAXWWMH', 'PAXSWMH']].values.astype(np.float32)
                    participant_buffers[seqn].append(arr)

        print(f"Finished reading Cycle {cycle} in {time.time()-t0:.1f}s. Assembling arrays ...")
        for seqn, chunks in participant_buffers.items():
            full_series = np.concatenate(chunks, axis=0)
            full_series = np.nan_to_num(full_series, nan=0.0)
            full_series = np.maximum(full_series, 0.0)
            
            n_rows = len(full_series)
            if n_rows >= T:
                cut = full_series[:T]
            else:
                # Pad to 168 if slightly shorter
                pad_len = T - n_rows
                pad_data = np.repeat(np.median(full_series, axis=0, keepdims=True), pad_len, axis=0)
                cut = np.vstack([full_series, pad_data])
                
            idx = seqn_to_idx[seqn]
            sequence_array[idx] = cut
            seqn_found.add(seqn)

    print(f"\nSuccessfully populated sequence tensors for {len(seqn_found)} / {N} participants.")

    # Static Feature Matrix (9 Vitals & Demographics)
    static_cols = [
        'resting_hr_bpm', 'bmi', 'waist_circ_cm', 'systolic_bp', 'diastolic_bp',
        'age', 'gender', 'ethnicity', 'poverty_ratio'
    ]
    X_static = cohort[static_cols].copy()
    for col in X_static.columns:
        if X_static[col].isna().sum() > 0:
            X_static[col] = X_static[col].fillna(X_static[col].median())
    static_array = X_static.values.astype(np.float32)

    # Targets
    class_mapping = {'Low_Risk_Normal': 0, 'Moderate_Risk_Prediabetes': 1, 'High_Risk_IR': 2}
    y_3class = cohort['metabolic_risk_class'].map(class_mapping).values.astype(np.int64)
    y_homa = cohort['homa_ir'].values.astype(np.float32)
    y_hba1c = cohort['hba1c_pct'].values.astype(np.float32)
    y_glucose = cohort['fasting_glucose_mgdl'].values.astype(np.float32)

    out_npz = os.path.join(DATA_DIR, 'nhanes_168h_sequence_tensors.npz')
    np.savez_compressed(
        out_npz,
        sequences=sequence_array,
        static_features=static_array,
        static_cols=np.array(static_cols),
        y_3class=y_3class,
        y_homa=y_homa,
        y_hba1c=y_hba1c,
        y_glucose=y_glucose,
        seqns=np.array(seqn_list)
    )
    print(f"Saved compressed sequence tensors to: {out_npz} ({os.path.getsize(out_npz)/(1024*1024):.2f} MB)")
    print(f"Sequence Tensor Shape: {sequence_array.shape} (Participants, 168 Hours, 3 Channels)")
    print(f"Static Feature Shape: {static_array.shape} (Participants, 9 Features)")

if __name__ == '__main__':
    build_sequences()
