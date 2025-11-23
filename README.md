# Whisper Fine-tuning for Hindi ASR

This project fine-tunes OpenAI's Whisper-small model on a Hindi ASR dataset and evaluates it on the FLEURS Hindi test set.

## Project Structure

```
joshtalks-whisper/
├── data/
│   ├── audio/              # Raw audio files
│   ├── transcript/         # Raw transcript JSON files
│   ├── metadata/           # Metadata JSON files
│   ├── processed_audio/    # Preprocessed audio files (16kHz, mono, trimmed)
│   ├── processed_transcript/ # Preprocessed transcript text files
│   ├── ft_data.csv         # Original dataset CSV
│   └── preprocessed_data.csv # Preprocessed dataset CSV
├── models/
│   └── whisper-small-hindi/ # Fine-tuned model (created after training)
├── results/                # Evaluation results (created after evaluation)
├── notebooks/
│   └── Preprocessing.ipynb # Exploratory preprocessing notebook
├── src/
│   ├── download_assets.py  # Download audio, transcripts, and metadata
│   ├── preprocess.py       # Preprocessing pipeline
│   ├── finetune_whisper.py # Fine-tuning script
│   └── evaluate.py         # Evaluation script
├── requirements.txt
└── README.md
```

## Documentation

📚 **Comprehensive Documentation Available:**
- **[Preprocessing Analysis](docs/PREPROCESSING_ANALYSIS.md)** - Detailed technical documentation with rationale, trade-offs, statistics, and best practices (Senior ML Engineer perspective)
- **[Executive Summary](docs/PREPROCESSING_EXECUTIVE_SUMMARY.md)** - High-level overview for quick reference
- **[Preprocessing Summary](PREPROCESSING_SUMMARY.md)** - Quick reference guide

## Dataset

The dataset contains ~10 hours of Hindi ASR training data with:
- Audio files (WAV format)
- Transcripts (JSON format with timestamps)
- Metadata (speaker information, device info, etc.)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd joshtalks-whisper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) If you need to download the dataset:
```bash
python src/download_assets.py
```

## Usage

### Step 1: Preprocessing

Preprocess the dataset to prepare it for training:

```bash
python src/preprocess.py
```

**What this does:**
- Converts audio to mono channel
- Resamples audio to 16kHz (Whisper's required sample rate)
- Normalizes audio amplitude
- Trims silence from beginning and end
- Normalizes transcript text (Unicode normalization, whitespace cleanup)
- Creates `data/preprocessed_data.csv` with paths to processed files

**Preprocessing details:**
- **Audio processing:**
  - Sample rate: 16kHz (Whisper standard)
  - Format: Mono WAV
  - Normalization: Amplitude normalized to [-1, 1]
  - Silence trimming: Top 30dB threshold
  
- **Text normalization:**
  - Unicode normalization (NFKC)
  - Whitespace normalization
  - Quote normalization
  - Character filtering (keeps Hindi Devanagari + basic punctuation)

### Step 2: Fine-tuning

Fine-tune Whisper-small on the preprocessed dataset:

```bash
python src/finetune_whisper.py
```

**Training configuration:**
- Model: `openai/whisper-small`
- Batch size: 8 (with gradient accumulation of 4)
- Learning rate: 1e-5
- Epochs: 3
- Warmup steps: 500
- Max audio duration: 30 seconds (Whisper's context limit)
- Evaluation: Every 500 steps
- Early stopping: Patience of 3

The fine-tuned model will be saved to `models/whisper-small-hindi/`.

### Step 3: Evaluation

Evaluate both baseline and fine-tuned models on FLEURS Hindi test set:

```bash
python src/evaluate.py
```

**What this does:**
- Loads FLEURS Hindi test dataset
- Evaluates baseline Whisper-small model
- Evaluates fine-tuned model (if available)
- Computes WER (Word Error Rate) and CER (Character Error Rate)
- Saves results to `results/evaluation_results.csv` and `results/evaluation_results.json`

**Output:**
- Structured table with WER and CER for both models
- Improvement metrics showing fine-tuning gains
- Detailed JSON file with predictions and references

## Results

After evaluation, results are saved in:
- `results/evaluation_results.csv` - Summary table
- `results/evaluation_results.json` - Detailed results with predictions

The summary table includes:
- Model name
- WER (Word Error Rate) in percentage
- CER (Character Error Rate) in percentage
- Number of test samples
- Improvement metrics (for fine-tuned vs baseline)

## Preprocessing Pipeline Details

### Audio Preprocessing

1. **Load audio**: Handles both mono and stereo files
2. **Convert to mono**: If stereo, convert to mono
3. **Resample**: Resample to 16kHz (Whisper's required sample rate)
4. **Normalize**: Normalize amplitude to [-1, 1] range
5. **Trim silence**: Remove leading/trailing silence (30dB threshold)
6. **Save**: Save as WAV format

### Text Preprocessing

1. **Unicode normalization**: NFKC normalization
2. **Whitespace cleanup**: Remove extra whitespace
3. **Quote normalization**: Normalize smart quotes
4. **Character filtering**: Keep only valid characters (Hindi Devanagari + basic punctuation)
5. **Final cleanup**: Strip leading/trailing whitespace

## Model Details

- **Base Model**: OpenAI Whisper-small
- **Parameters**: ~244M
- **Language**: Hindi (hi)
- **Task**: Automatic Speech Recognition (ASR)
- **Input**: 16kHz mono audio
- **Output**: Hindi text transcription

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- Datasets >= 2.14.0
- Librosa >= 0.10.0
- SoundFile >= 0.12.0
- Evaluate >= 0.4.0

## Notes

- The preprocessing script filters out audio files longer than 30 seconds (Whisper's context limit)
- Training uses mixed precision (FP16) on GPU for faster training
- Early stopping is enabled to prevent overfitting
- The evaluation script automatically downloads the FLEURS Hindi test set

## Troubleshooting

1. **Out of memory errors**: Reduce `BATCH_SIZE` or `GRADIENT_ACCUMULATION_STEPS` in `finetune_whisper.py`
2. **Audio loading errors**: Ensure audio files are in WAV format and not corrupted
3. **CUDA errors**: Check GPU availability and CUDA installation
4. **Dataset download issues**: Check internet connection and HuggingFace access

## License

[Add your license information here]

## Acknowledgments

- OpenAI for the Whisper model
- HuggingFace for Transformers and Datasets libraries
- Google for the FLEURS dataset

