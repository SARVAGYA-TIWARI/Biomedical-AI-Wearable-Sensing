import pandas as pd

ecg_sample = r"d:\BTP\diabetes_subset_ecg_data\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_ECG.csv"
df = pd.read_csv(ecg_sample, nrows=20)
print("=== First 20 rows of ECG.csv ===")
print(df)
print("\nData types:")
print(df.dtypes)
