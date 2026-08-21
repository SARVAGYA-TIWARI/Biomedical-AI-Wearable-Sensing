import json
import os

with open(r"d:\BTP\01_D1NAMO_Glucose_EDA_Baseline (1) (1).ipynb", 'r', encoding='utf-8', errors='ignore') as f:
    nb1 = json.load(f)

with open(r"d:\BTP\02_D1NAMO_Phase1_Multimodal_Modeling (1).ipynb", 'r', encoding='utf-8', errors='ignore') as f:
    nb2 = json.load(f)

print("=== NB1 Cells ===")
for i in range(min(15, len(nb1['cells']))):
    c = nb1['cells'][i]
    src = "".join(c.get('source', []))
    print(f"Cell {i} ({c['cell_type']}):\n{src[:200]}\n")

print("\n=== NB2 Cells ===")
for i in range(min(15, len(nb2['cells']))):
    c = nb2['cells'][i]
    src = "".join(c.get('source', []))
    print(f"Cell {i} ({c['cell_type']}):\n{src[:200]}\n")
