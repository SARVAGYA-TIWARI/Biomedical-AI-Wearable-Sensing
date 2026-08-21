import json

with open(r"d:\BTP\01_D1NAMO_Glucose_EDA_Baseline (1) (1).ipynb", 'r', encoding='utf-8', errors='ignore') as f:
    nb = json.load(f)

for idx in range(12):
    cell = nb['cells'][idx]
    src = "".join(cell.get('source', []))
    print(f"=== Cell {idx} ({cell.get('cell_type')}) ===")
    print(src)
