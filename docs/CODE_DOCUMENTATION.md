# Source Code Documentation

## Overview

This document provides comprehensive documentation for all source code in the `src/` directory. Each module is analyzed from a machine learning engineering perspective, explaining design decisions, implementation details, and technical rationale.

## Directory Structure

```
src/
├── download_assets.py    # Data acquisition module
├── preprocess.py         # Data preprocessing pipeline
├── finetune_whisper.py   # Model fine-tuning module
└── evaluate.py           # Model evaluation module
```

## Module: download_assets.py

### Purpose

Downloads audio files, transcripts, and metadata from remote URLs specified in a CSV file. This module handles the data acquisition phase of the pipeline.

### Architecture

The module follows a simple sequential download pattern:
1. Reads CSV file with URLs
2. Extracts file identifiers from URLs
3. Downloads files to local directories
4. Organizes files by type (audio, transcript, metadata)

### Key Functions

#### `get_id_from_url(url: str) -> str`

**Purpose:** Extracts a unique identifier from a URL path.

**Implementation:**
```python
def get_id_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path)
    return filename.split('_')[0] if '_' in filename else filename.split('.')[0]
```

**Technical Details:**
- Uses `urlparse` to extract path component from URL
- Gets filename from path using `os.path.basename()`
- Handles two naming conventions:
  - Files with underscores: takes prefix before first underscore
  - Files without underscores: takes prefix before first dot (extension)

**Design Rationale:**
This approach assumes consistent URL naming conventions. Real-world scenarios may require more robust parsing, but this simple heuristic works for structured datasets.

**Limitations:**
- Assumes identifier is prefix of filename
- May fail with complex filename patterns
- No validation of extracted ID format

#### `download_file(url: str, dest_path: str) -> None`

**Purpose:** Downloads a file from a URL and saves it locally.

**Implementation:**
```python
def download_file(url, dest_path):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(r.content)
```

**Technical Details:**
- Uses `requests` library for HTTP operations
- 30-second timeout prevents hanging on slow/unresponsive servers
- `raise_for_status()` raises exception for HTTP error codes (4xx, 5xx)
- Binary mode writing (`'wb'`) ensures correct file handling

**Error Handling:**
- Network errors: Propagated as exceptions
- HTTP errors: Raised via `raise_for_status()`
- File I/O errors: Propagated as exceptions

**Design Considerations:**
- No retry logic: Single attempt per file
- No resume capability: Failed downloads must restart
- No progress reporting: Silent execution
- Synchronous downloads: Sequential processing

**Performance:**
For large datasets, sequential downloads can be slow. Parallel downloads would improve performance but increase complexity and server load.

#### `main() -> None`

**Purpose:** Orchestrates the download process for all files in the CSV.

**Implementation Flow:**
1. Opens CSV file with UTF-8 encoding
2. Iterates through rows using `csv.DictReader`
3. For each row:
   - Extracts URLs for audio, transcript, and metadata
   - Derives file IDs and paths
   - Downloads each file type

**File Organization:**
- Audio files: `data/audio/{id}.{ext}`
- Transcript files: `data/transcript/{id}.{ext}`
- Metadata files: `data/metadata/{id}.{ext}`

**Error Propagation:**
If any download fails, the exception propagates and the script terminates. This is appropriate for batch operations where partial downloads may be problematic.

**CSV Schema Assumption:**
The code assumes CSV columns:
- `rec_url_gcp`: Audio file URL
- `transcription_url_gcp`: Transcript file URL
- `metadata_url_gcp`: Metadata file URL

### Usage

```bash
python src/download_assets.py
```

**Prerequisites:**
- CSV file at `data/ft_data.csv`
- Network connectivity to download URLs
- Sufficient disk space for downloaded files

### Limitations

1. **No Progress Tracking:** Silent execution provides no feedback
2. **No Resume:** Interrupted downloads cannot be resumed
3. **No Validation:** Downloaded files are not verified for correctness
4. **No Parallelization:** Sequential downloads are slow for large datasets
5. **No Error Recovery:** Single failure stops entire process

## Module: preprocess.py

### Purpose

Preprocesses raw audio and transcript files to prepare them for model training. This is the most critical preprocessing step, ensuring data quality and format consistency.

### Architecture

The preprocessing pipeline consists of:
1. Audio preprocessing: Format conversion, resampling, normalization
2. Text preprocessing: Unicode normalization, filtering, cleaning
3. Metadata generation: CSV file with processed file paths

### Key Constants

#### `TARGET_SR = 16000`

Target sample rate in Hz. Whisper models expect 16kHz audio. This matches the training distribution of Whisper models.

#### `TRIM_TOP_DB = 30`

Silence trimming threshold in decibels. Lower values trim more aggressively. 30 dB is a balanced choice that removes significant silence while preserving quiet speech.

### Key Functions

#### `normalize_text(text: str) -> str`

**Purpose:** Normalizes Hindi text for ASR training.

**Processing Steps:**
1. Unicode normalization (NFKC)
2. Quote normalization (smart quotes → straight quotes)
3. Whitespace normalization (multiple spaces → single space)
4. Character filtering (keeps only Devanagari + basic punctuation)

See `PREPROCESSING_TECHNICAL.md` for detailed analysis of text normalization.

#### `extract_full_transcript(transcript_path: str) -> str`

**Purpose:** Extracts and normalizes transcript from JSON file.

**Implementation:**
- Parses JSON file containing timestamped segments
- Concatenates segment texts with spaces
- Applies text normalization

**JSON Schema Assumption:**
Array of objects with 'text' field:
```json
[{"text": "segment 1"}, {"text": "segment 2"}]
```

#### `process_audio(audio_path: str, output_path: str) -> float`

**Purpose:** Processes audio file for Whisper training.

**Returns:** Duration of processed audio in seconds.

**Processing Steps:**
1. Load audio (preserves original sample rate)
2. Convert stereo to mono if needed
3. Resample to 16kHz if needed
4. Normalize amplitude to [-1, 1]
5. Trim leading/trailing silence
6. Save as 16kHz mono WAV

See `PREPROCESSING_TECHNICAL.md` for detailed analysis of audio processing.

#### `main() -> None`

**Purpose:** Main preprocessing pipeline.

**Workflow:**
1. Read input CSV (`ft_data.csv`)
2. For each recording:
   - Validate file existence
   - Process audio file
   - Extract and normalize transcript
   - Create metadata record
3. Write output CSV (`preprocessed_data.csv`)

**Output CSV Schema:**
- `recording_id`: Unique identifier
- `user_id`: Speaker identifier
- `language`: Language code
- `original_duration`: Original audio duration
- `processed_duration`: Processed audio duration
- `audio_path`: Path to processed audio
- `transcript_path`: Path to processed transcript
- `transcript`: Full transcript text

### Usage

```bash
python src/preprocess.py
```

**Prerequisites:**
- Raw audio files in `data/audio/`
- Raw transcript files in `data/transcript/`
- Input CSV at `data/ft_data.csv`

**Output:**
- Processed audio in `data/processed_audio/`
- Processed transcripts in `data/processed_transcript/`
- Metadata CSV at `data/preprocessed_data.csv`

## Module: finetune_whisper.py

### Purpose

Fine-tunes the Whisper-small model on the preprocessed Hindi ASR dataset. This module implements the training loop, data preparation, and model checkpointing.

### Architecture

The fine-tuning process follows the Hugging Face Transformers training paradigm:
1. Load base model and processor
2. Prepare dataset from CSV
3. Configure training arguments
4. Create trainer with custom data collator
5. Train model with evaluation and checkpointing

### Key Constants

#### `MODEL_NAME = "openai/whisper-small"`

Base model identifier from Hugging Face Hub. Whisper-small has ~244M parameters and provides a good balance between performance and training efficiency.

#### `BATCH_SIZE = 4`

Number of samples per training batch. Combined with gradient accumulation, effective batch size is 16.

#### `GRADIENT_ACCUMULATION_STEPS = 4`

Number of gradient accumulation steps. Effective batch size = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS = 16.

**Rationale:** Allows large effective batch sizes without exceeding GPU memory limits.

#### `LEARNING_RATE = 1e-5`

Initial learning rate for Adam optimizer. This is a conservative learning rate appropriate for fine-tuning pretrained models.

**Rationale:** Lower learning rates prevent catastrophic forgetting while allowing task-specific adaptation.

#### `NUM_EPOCHS = 3`

Number of training epochs. Limited epochs prevent overfitting on small datasets.

#### `WARMUP_STEPS = 500`

Number of warmup steps for learning rate scheduler. During warmup, learning rate increases linearly from 0 to target rate.

**Rationale:** Gradual learning rate ramp-up stabilizes training early stages.

#### `MAX_LABEL_LENGTH = 448`

Maximum length of tokenized transcript labels. Whisper tokenizer produces sequences of this length for typical speech samples.

### Key Classes

#### `DataCollatorSpeechSeq2SeqWithPadding`

**Purpose:** Custom data collator for batching variable-length audio and text sequences.

**Implementation Details:**
- Pads audio features using processor's feature extractor
- Pads text labels using processor's tokenizer
- Masks padding tokens in labels (set to -100 to ignore in loss)
- Ensures labels are on same device as inputs

**Technical Considerations:**
- Padding enables batch processing of variable-length sequences
- Masking prevents padding tokens from contributing to loss
- Device placement ensures compatibility with GPU training

**Why Custom Collator:**
Hugging Face's default collators may not handle audio feature padding correctly. This custom collator ensures proper alignment of audio features and text labels.

### Key Functions

#### `prepare_dataset(csv_path: str) -> Dataset`

**Purpose:** Loads dataset from preprocessed CSV file.

**Processing:**
1. Reads CSV file
2. Validates required columns exist
3. Checks audio files exist on disk
4. Creates Hugging Face Dataset from records

**Error Handling:**
- Missing columns: Row skipped
- Missing files: Row skipped
- Empty dataset: Raises ValueError

**Dataset Structure:**
Each record contains:
- `audio`: Path to audio file
- `text`: Transcript text

#### `prepare_dataset_batch(example: dict, processor, max_label_length: int) -> dict`

**Purpose:** Prepares a single example for training by extracting features.

**Processing:**
1. Loads audio file using Librosa (16kHz, mono)
2. Extracts audio features using Whisper feature extractor
3. Tokenizes transcript text
4. Truncates labels to max length

**Feature Extraction:**
Whisper's feature extractor converts raw audio to mel-spectrogram features. This is done on-the-fly during dataset mapping to avoid storing large feature arrays.

**Tokenization:**
Whisper tokenizer converts text to token IDs. Truncation ensures labels fit within model's sequence length limits.

#### `compute_metrics(pred: EvalPrediction) -> dict`

**Purpose:** Computes Word Error Rate (WER) during evaluation.

**Implementation:**
1. Decodes predictions and labels from token IDs
2. Computes WER using `evaluate` library
3. Returns metrics dictionary

**Why WER:**
WER is the standard metric for ASR tasks, measuring transcription accuracy at word level.

**Evaluation Frequency:**
Computed during validation steps (every 500 steps) to monitor training progress.

#### `main() -> None`

**Purpose:** Main training pipeline.

**Training Workflow:**
1. Initialize device (CUDA if available, else CPU)
2. Load model and processor
3. Prepare dataset from CSV
4. Filter long samples (text length < 4500 characters)
5. Map dataset to extract features
6. Filter samples with excessive labels
7. Split into train/validation sets (90/10)
8. Create data collator
9. Configure training arguments
10. Create trainer
11. Train model
12. Save final model

**Training Configuration:**
- Mixed precision (FP16) on GPU for faster training
- Evaluation every 500 steps
- Checkpoint saving every 500 steps
- Early stopping with patience of 3
- TensorBoard logging

**Model Saving:**
- Model weights saved to `models/whisper-small-hindi/`
- Processor (tokenizer + feature extractor) saved with model
- Checkpoints saved during training for recovery

### Usage

```bash
python src/finetune_whisper.py
```

**Prerequisites:**
- Preprocessed CSV at `data/preprocessed_data.csv`
- Processed audio files accessible
- Sufficient GPU memory (recommended) or CPU with patience
- CUDA installed for GPU training

**Output:**
- Fine-tuned model in `models/whisper-small-hindi/`
- Training checkpoints
- TensorBoard logs

**Training Time:**
Depends on dataset size and hardware:
- GPU (V100/A100): Several hours for ~10 hours of audio
- CPU: Days for same dataset

## Module: evaluate.py

### Purpose

Evaluates both baseline and fine-tuned Whisper models on the FLEURS Hindi test dataset. Provides comprehensive metrics including WER and CER.

### Architecture

The evaluation process:
1. Loads FLEURS Hindi test dataset
2. Evaluates baseline model
3. Evaluates fine-tuned model (if available)
4. Computes comparison metrics
5. Saves results to CSV and JSON

### Key Constants

#### `BASELINE_MODEL = "openai/whisper-small"`

Baseline model identifier. This is the pretrained model without fine-tuning, used as a comparison point.

#### `FINE_TUNED_MODEL`

Path to fine-tuned model directory. Defaults to `models/whisper-small-hindi/`.

#### `RESULTS_CSV` and `RESULTS_JSON`

Output file paths for evaluation results:
- `RESULTS_CSV`: Path to `results/FT Result.csv` - Contains summary table with WER and CER metrics
- `RESULTS_JSON`: Path to `results/FT Result.json` - Contains detailed results including sample predictions and references

### Key Functions

#### `load_fleurs_hindi_test() -> Dataset`

**Purpose:** Loads FLEURS Hindi test dataset from Hugging Face Hub.

**Implementation:**
```python
dataset = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

**Technical Details:**
- Loads Hindi (hi_in) configuration of FLEURS dataset
- Uses test split for evaluation
- `trust_remote_code=True` allows custom dataset loading scripts
- Casts audio column to 16kHz for consistency

**Dataset Characteristics:**
- Standard benchmark for multilingual ASR
- High-quality transcriptions
- Diverse speech conditions
- Reproducible evaluation

#### `transcribe_batch(model, processor, audio_arrays: list, device: str, batch_size: int = 8) -> list`

**Purpose:** Transcribes batches of audio using the Whisper model.

**Implementation:**
1. Processes audio in batches
2. Extracts features using processor
3. Generates transcriptions using model.generate()
4. Decodes token IDs to text

**Batch Processing:**
- Batch size of 8 balances memory usage and speed
- Larger batches speed up inference but require more GPU memory
- Smaller batches reduce memory but slow down processing

**Generation Configuration:**
- Language: Hindi (hi)
- Task: Transcribe (not translate)
- Beam search or greedy decoding (model default)

#### `evaluate_model(model_path: str, model_name: str, dataset: Dataset, device: str) -> dict`

**Purpose:** Evaluates a Whisper model on the test dataset.

**Evaluation Process:**
1. Loads model and processor from path
2. Extracts audio arrays and reference transcriptions
3. Generates predictions using batch transcription
4. Computes WER and CER metrics
5. Returns results dictionary

**Metrics:**
- **WER (Word Error Rate):** Percentage of words incorrectly transcribed
- **CER (Character Error Rate):** Percentage of characters incorrectly transcribed

**Results Structure:**
- Model metadata (name, path)
- Number of samples
- WER and CER scores
- Sample predictions and references (first 10)

#### `main() -> None`

**Purpose:** Main evaluation pipeline.

**Evaluation Workflow:**
1. Detects available device (CUDA/CPU)
2. Loads FLEURS Hindi test dataset
3. Evaluates baseline model
4. Evaluates fine-tuned model (if exists)
5. Computes improvement metrics
6. Generates formatted results table
7. Saves results to CSV and JSON

**Results Output:**
- Console: Formatted table with metrics
- CSV: `results/FT Result.csv` - Summary table
- JSON: `results/FT Result.json` - Detailed results

**Improvement Metrics:**
Computes relative improvement:
- WER improvement: `((baseline_wer - fine_tuned_wer) / baseline_wer) × 100`
- CER improvement: `((baseline_cer - fine_tuned_cer) / baseline_cer) × 100`

### Usage

```bash
python src/evaluate.py
```

**Prerequisites:**
- Internet connection (for downloading FLEURS dataset)
- Baseline model accessible from Hugging Face Hub
- Fine-tuned model in `models/whisper-small-hindi/` (optional)

**Output:**
- Console output with formatted results
- CSV file with summary table
- JSON file with detailed results including sample predictions

**Evaluation Time:**
Depends on test set size and hardware:
- GPU: Minutes to hours depending on dataset size
- CPU: Significantly longer

## Design Patterns and Best Practices

### Modularity

Each module has a single, well-defined responsibility:
- `download_assets.py`: Data acquisition
- `preprocess.py`: Data preprocessing
- `finetune_whisper.py`: Model training
- `evaluate.py`: Model evaluation

This separation enables:
- Independent testing
- Easy modification
- Clear understanding of pipeline stages

### Error Handling

**Current Approach:**
- Minimal error handling
- Exceptions propagate to caller
- Fail-fast philosophy

**Rationale:**
For batch processing scripts, early failure identification is preferred over silent error handling. This approach makes debugging easier.

**Trade-offs:**
More robust error handling could improve user experience but adds complexity. Current approach prioritizes simplicity and clarity.

### Configuration

**Current Approach:**
- Constants defined at module level
- Hard-coded paths and parameters

**Rationale:**
Simple configuration is sufficient for focused use cases. Complex configuration systems add overhead without clear benefit.

**Future Considerations:**
For production deployment, consider:
- Configuration files (YAML, JSON)
- Command-line arguments
- Environment variables

### Reproducibility

All modules are designed for reproducibility:
- Deterministic processing (no randomness in preprocessing)
- Fixed random seeds (training uses seed=42)
- Versioned dependencies (requirements.txt)

## Performance Considerations

### Preprocessing Performance

**Bottlenecks:**
- Audio resampling: CPU-intensive
- File I/O: Disk speed dependent

**Optimization Opportunities:**
- Parallel processing with multiprocessing
- Incremental processing (skip completed files)
- Batch audio processing

### Training Performance

**Bottlenecks:**
- Feature extraction: CPU-bound
- Model forward/backward passes: GPU-bound

**Optimization Opportunities:**
- Pre-extract features (trade-off: storage vs. speed)
- Gradient checkpointing (already available)
- Mixed precision training (already enabled)

### Evaluation Performance

**Bottlenecks:**
- Model inference: GPU-bound
- Batch processing: Memory vs. speed trade-off

**Optimization Opportunities:**
- Larger batch sizes (if memory permits)
- Model quantization for faster inference
- Distributed evaluation for large datasets

## Testing Recommendations

### Unit Testing

Each function should have unit tests:
- `normalize_text()`: Test various Unicode scenarios
- `process_audio()`: Test different audio formats
- `prepare_dataset_batch()`: Test feature extraction

### Integration Testing

End-to-end pipeline tests:
- Download → Preprocess → Train → Evaluate
- Verify output formats and quality

### Validation Testing

Data quality checks:
- Verify processed audio format
- Validate transcript normalization
- Check dataset statistics

## Conclusion

The source code modules form a complete pipeline for fine-tuning Whisper on Hindi ASR data. The design prioritizes simplicity, correctness, and maintainability. Each module serves a clear purpose and can be understood and modified independently. The codebase provides a solid foundation for ASR fine-tuning projects.

