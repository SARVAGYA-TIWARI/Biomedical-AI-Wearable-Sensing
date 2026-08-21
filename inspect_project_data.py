import json
import os
import glob

def summarize_notebook(nb_path):
    print(f"\n==========================================")
    print(f"SUMMARY OF NOTEBOOK: {nb_path}")
    print(f"==========================================")
    with open(nb_path, 'r', encoding='utf-8', errors='ignore') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    print(f"Total cells: {len(cells)}")
    for idx, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        if cell_type == 'markdown':
            lines = [line.strip() for line in source.split('\n') if line.strip().startswith('#')]
            if lines:
                print(f"  [Cell {idx} MD]: {', '.join(lines[:3])}")
        elif cell_type == 'code':
            # Check for key function definitions, modeling steps, or comments
            first_line = source.strip().split('\n')[0] if source.strip() else ""
            if any(k in source.lower() for k in ['model', 'xgboost', 'lstm', 'glucose', 'mae', 'rmse', 'evaluate', 'train', 'leave_one_out', 'loso', 'split']):
                print(f"  [Cell {idx} CODE]: {first_line[:80]}...")

def inspect_dataset_tree(root_dir):
    print(f"\n==========================================")
    print(f"DATASET STRUCTURE: {root_dir}")
    print(f"==========================================")
    for root, dirs, files in os.walk(root_dir):
        depth = root.replace(root_dir, '').count(os.sep)
        if depth <= 4:
            indent = '  ' * depth
            print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")
            if files and depth >= 2:
                for f in files[:10]:
                    size = os.path.getsize(os.path.join(root, f))
                    print(f"{indent}  - {f} ({size / (1024*1024):.2f} MB)")
                if len(files) > 10:
                    print(f"{indent}  ... and {len(files)-10} more files")

if __name__ == '__main__':
    summarize_notebook(r'd:\BTP\01_D1NAMO_Glucose_EDA_Baseline (1) (1).ipynb')
    summarize_notebook(r'd:\BTP\02_D1NAMO_Phase1_Multimodal_Modeling (1).ipynb')
    inspect_dataset_tree(r'd:\BTP\diabetes_subset_ecg_data')
