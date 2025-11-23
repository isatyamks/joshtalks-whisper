"""
Main pipeline script to run the complete workflow.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(1)

def main():
    data_csv = Path("data/ft_data.csv")
    if not data_csv.exists():
        sys.exit(1)

    run_command("python src/preprocess.py", "Preprocessing dataset")

    response = input("Run evaluation on FLEURS test set? (y/n): ")
    if response.lower() == 'y':
        run_command("python src/evaluate.py", "Evaluating models")

if __name__ == "__main__":
    main()
