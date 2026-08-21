import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generate_comprehensive_report import create_report

if __name__ == '__main__':
    print("==========================================================================")
    print("Generating Comprehensive Research Report (.docx)...")
    print("==========================================================================")
    create_report()
