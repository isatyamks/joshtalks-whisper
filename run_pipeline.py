"""
Main pipeline script to run the complete workflow:
1. Preprocessing
2. Fine-tuning
3. Evaluation
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    else:
        print(f"\n✅ {description} completed successfully!")

def main():
    """Run the complete pipeline."""
    print("="*60)
    print("Whisper Fine-tuning Pipeline")
    print("="*60)
    
    # Check if data exists
    data_csv = Path("data/ft_data.csv")
    if not data_csv.exists():
        print("❌ Error: data/ft_data.csv not found!")
        print("   Please ensure the dataset CSV file exists.")
        sys.exit(1)
    
    # Step 1: Preprocessing
    print("\n📋 Step 1/3: Preprocessing")
    run_command(
        "python src/preprocess.py",
        "Preprocessing dataset"
    )
    
    # # Check if preprocessing was successful
    # preprocessed_csv = Path("data/preprocessed_data.csv")
    # if not preprocessed_csv.exists():
    #     print("❌ Error: Preprocessing did not create preprocessed_data.csv")
    #     sys.exit(1)
    
    # # Step 2: Fine-tuning
    # print("\n📋 Step 2/3: Fine-tuning")
    # response = input("\nDo you want to run fine-tuning? This may take several hours. (y/n): ")
    # if response.lower() == 'y':
    #     run_command(
    #         "python src/finetune_whisper.py",
    #         "Fine-tuning Whisper-small"
    #     )
    # else:
    #     print("⏭️  Skipping fine-tuning")
    
    # Step 3: Evaluation
    print("\n📋 Step 3/3: Evaluation")
    response = input("\nDo you want to run evaluation on FLEURS test set? (y/n): ")
    if response.lower() == 'y':
        run_command(
            "python src/evaluate.py",
            "Evaluating models on FLEURS test set"
        )
    else:
        print("⏭️  Skipping evaluation")
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)
    print("\nNext steps:")
    print("  - Check results/ directory for evaluation results")
    print("  - Check models/ directory for fine-tuned model")
    print("  - Review PREPROCESSING_SUMMARY.md for preprocessing details")

if __name__ == "__main__":
    main()

