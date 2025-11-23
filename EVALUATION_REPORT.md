# Evaluation Report: Whisper Models on FLEURS Hindi Test Set

## Overview

This document reports the evaluation results for both the baseline Whisper-small model and the fine-tuned Whisper-small model on the Hindi portion of the FLEURS test dataset.

---

## Evaluation Metrics

### Word Error Rate (WER)
- **Definition**: Percentage of words incorrectly transcribed
- **Formula**: `WER = (Substitutions + Insertions + Deletions) / Total Words × 100`
- **Lower is better**: WER of 0% means perfect transcription

### Character Error Rate (CER)
- **Definition**: Percentage of characters incorrectly transcribed
- **Formula**: `CER = (Substitutions + Insertions + Deletions) / Total Characters × 100`
- **Lower is better**: CER of 0% means perfect transcription

---

## Test Dataset

- **Dataset**: FLEURS (FREsh and Useful Resources for Speech recognition)
- **Language**: Hindi (hi_in)
- **Split**: Test set
- **Source**: Google FLEURS dataset via Hugging Face Datasets

---

## Models Evaluated

1. **Baseline Model**: `openai/whisper-small`
   - Pre-trained Whisper-small model
   - No fine-tuning on Hindi data
   - Used as baseline for comparison

2. **Fine-tuned Model**: `models/whisper-small-hindi`
   - Fine-tuned Whisper-small on Hindi ASR dataset
   - Trained on ~10 hours of Hindi speech data
   - Fine-tuning configuration:
     - Learning rate: 1e-5
     - Epochs: 3
     - Batch size: 8 (with gradient accumulation of 4)

---

## Results

### Evaluation Results Table

| Model | WER (%) | CER (%) | Num Samples |
|-------|---------|---------|-------------|
| Whisper-small (Baseline) | [Value] | [Value] | [Value] |
| Whisper-small (Fine-tuned) | [Value] | [Value] | [Value] |
| Improvement | [Value]% | [Value]% | - |

*Note: Run `python src/evaluate.py` to generate actual results*

### Performance Summary

- **Baseline WER**: [Will be populated after evaluation]
- **Fine-tuned WER**: [Will be populated after evaluation]
- **WER Improvement**: [Will be populated after evaluation]%
- **Baseline CER**: [Will be populated after evaluation]
- **Fine-tuned CER**: [Will be populated after evaluation]
- **CER Improvement**: [Will be populated after evaluation]%

---

## Detailed Results

Detailed evaluation results including predictions and references are saved in:
- **CSV Format**: `results/evaluation_results.csv`
- **JSON Format**: `results/evaluation_results.json`

The JSON file contains:
- Model metadata
- WER and CER scores
- Sample predictions and references
- Full evaluation statistics

---

## Running Evaluation

### Command
```bash
python src/evaluate.py
```

### What It Does
1. Loads FLEURS Hindi test dataset
2. Evaluates baseline Whisper-small model
3. Evaluates fine-tuned model (if available)
4. Computes WER and CER metrics
5. Generates structured results table
6. Saves results to CSV and JSON files

### Output
- Console output with formatted results table
- `results/evaluation_results.csv` - Summary table
- `results/evaluation_results.json` - Detailed results

---

## Evaluation Process

1. **Dataset Loading**
   - Load FLEURS Hindi test split
   - Configure audio to 16kHz sample rate
   - Prepare audio arrays and reference transcriptions

2. **Model Inference**
   - Load model and processor
   - Process audio in batches
   - Generate transcriptions using Whisper decoder
   - Language set to Hindi (hi), task set to transcribe

3. **Metric Computation**
   - Compute WER using `evaluate` library
   - Compute CER using `evaluate` library
   - Compare predictions against references

4. **Results Generation**
   - Format results in structured table
   - Calculate improvement metrics
   - Save to files

---

## Notes

- Evaluation runs on CPU or GPU automatically (uses CUDA if available)
- Batch size for inference: 8 samples
- Language prompt: Hindi (hi)
- Task: Transcribe (not translate)

---

## Next Steps

After evaluation:
1. Review results in `results/evaluation_results.csv`
2. Analyze detailed predictions in `results/evaluation_results.json`
3. Compare baseline vs fine-tuned performance
4. Identify areas for further improvement

---

## References

- FLEURS Dataset: https://huggingface.co/datasets/google/fleurs
- Whisper Model: https://github.com/openai/whisper
- Evaluate Library: https://huggingface.co/docs/evaluate/index

