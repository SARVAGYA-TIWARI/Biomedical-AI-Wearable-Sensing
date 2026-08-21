import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phase1_benchmark_loso import run_phase1_loso_benchmark

if __name__ == '__main__':
    print("==========================================================================")
    print("Starting Phase 1: Benchmark Glucose Forecasting (LOSO Cross-Validation)")
    print("==========================================================================")
    run_phase1_loso_benchmark()
