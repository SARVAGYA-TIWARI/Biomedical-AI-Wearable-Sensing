import os
import sys

def inspect():
    print("Python exe:", sys.executable)
    for pkg in ['torch', 'torchvision', 'sklearn', 'xgboost', 'lightgbm', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'pypdf', 'fitz', 'docx', 'pptx', 'kaggle']:
        try:
            m = __import__(pkg)
            ver = getattr(m, '__version__', 'installed')
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: NOT installed")

if __name__ == '__main__':
    inspect()
