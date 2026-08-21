import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phase2_noninvasive_loso import run_phase2_noninvasive_benchmark

if __name__ == '__main__':
    print("==========================================================================")
    print("Starting Phase 2: Non-Invasive Glucose Prediction (LOSO Cross-Validation)")
    print("==========================================================================")
    run_phase2_noninvasive_benchmark()
