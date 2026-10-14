"""
Root entry point for Experiment A: Normal vs LDoS in Time Domain.
Executes the comprehensive evaluation pipeline defined in experiments/experiment_a.py.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.experiment_a import run_experiment_a

if __name__ == "__main__":
    run_experiment_a()
