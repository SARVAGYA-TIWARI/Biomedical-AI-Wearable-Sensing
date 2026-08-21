import json

def inspect_code_cells(nb_path):
    print(f"\n==========================================")
    print(f"CODE CELLS IN: {nb_path}")
    print(f"==========================================")
    with open(nb_path, 'r', encoding='utf-8', errors='ignore') as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            print(f"\n--- Cell {idx} ---")
            print(source[:500])

if __name__ == '__main__':
    inspect_code_cells(r'd:\BTP\01_D1NAMO_Glucose_EDA_Baseline (1) (1).ipynb')
    inspect_code_cells(r'd:\BTP\02_D1NAMO_Phase1_Multimodal_Modeling (1).ipynb')
