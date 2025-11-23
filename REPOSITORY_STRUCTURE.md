# Repository Structure for GitHub

## Files Tracked in Git

### Source Code
- `src/download_assets.py` - Data acquisition script
- `src/preprocess.py` - Preprocessing pipeline
- `src/finetune_whisper.py` - Model fine-tuning script
- `src/evaluate.py` - Model evaluation script

### Documentation
- `README.md` - Main project documentation
- `CONTRIBUTING.md` - Contributing guidelines
- `docs/` - Comprehensive technical documentation
  - `CODE_DOCUMENTATION.md` - Source code documentation
  - `PREPROCESSING_TECHNICAL.md` - Preprocessing technical analysis
  - `PREPROCESSING_ANALYSIS.md` - Detailed preprocessing analysis
  - `PREPROCESSING_EXECUTIVE_SUMMARY.md` - Executive summary
  - `README.md` - Documentation index

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore patterns

### Reports (Documentation)
- `PREPROCESSING_REPORT.md` - Preprocessing report
- `PREPROCESSING_SUMMARY.md` - Preprocessing summary
- `EVALUATION_REPORT.md` - Evaluation documentation
- `TASK_COMPLETION_REPORT.md` - Task completion summary

### Notebooks
- `notebooks/Preprocessing.ipynb` - Exploratory preprocessing notebook

### Directory Structure Placeholders
- `data/.gitkeep` - Keeps data directory in Git (contents ignored)
- `models/.gitkeep` - Keeps models directory in Git (contents ignored)
- `results/.gitkeep` - Keeps results directory in Git (contents ignored)

## Files Excluded from Git

The following are ignored by `.gitignore`:

### Data Files
- All audio files (`.wav`, `.mp3`, `.flac`)
- All transcript files (`.json`, `.txt`)
- All metadata files
- All CSV files in `data/` directory
- Processed audio and transcript files

### Model Files
- All model checkpoints (`.safetensors`, `.bin`, `.pt`, `.pth`)
- All files in `models/` directory

### Results
- All files in `results/` directory

### Cache and Temporary Files
- `__pycache__/` directories
- `.pyc` files
- Virtual environments
- IDE configuration files
- Jupyter notebook checkpoints
- Log files
- TensorBoard logs

### Large Binary Files
- PDF files
- Audio files (already listed above)
- Model weights (already listed above)

## Repository Size

With `.gitignore` in place, the repository will contain:
- Source code: ~10-20 KB
- Documentation: ~200-300 KB
- Configuration: ~1 KB
- Total tracked size: ~300-400 KB

Large files (models, data, results) are excluded to keep the repository lightweight and GitHub-friendly.

