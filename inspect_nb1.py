import json

with open(r"d:\BTP\01_D1NAMO_Glucose_EDA_Baseline (1) (1).ipynb", 'r', encoding='utf-8', errors='ignore') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get('cells', [])):
    ct = cell.get('cell_type')
    src = "".join(cell.get('source', []))
    print(f"[{idx} {ct}] {src[:300]}\n{'-'*40}")
