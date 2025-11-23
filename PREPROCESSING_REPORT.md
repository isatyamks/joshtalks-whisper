# Preprocessing Report: Hindi ASR Dataset

## Executive Summary

This document describes the preprocessing pipeline applied to prepare the Hindi ASR dataset for fine-tuning the Whisper-small model. The preprocessing ensures all audio and text data conforms to Whisper's requirements and best practices for ASR training.

---

## Preprocessing Overview

The preprocessing pipeline consists of two main components:
1. **Audio Preprocessing**: Converts raw audio files to Whisper-compatible format
2. **Text Preprocessing**: Normalizes transcriptions for consistent training data

---

## Audio Preprocessing Steps

### 1. Audio Loading
- **Purpose**: Load audio files preserving original sample rate
- **Method**: Uses `librosa.load()` which handles multiple audio formats
- **Handles**: Both mono and stereo audio files

### 2. Mono Conversion
- **Purpose**: Ensure consistent single-channel audio
- **Method**: Converts stereo to mono using `librosa.to_mono()`
- **Rationale**: Whisper processes mono audio, reduces data redundancy

### 3. Resampling to 16kHz
- **Purpose**: Match Whisper's required sample rate
- **Method**: High-quality resampling using `librosa.resample()`
- **Input**: Various sample rates (typically 44.1kHz or 48kHz)
- **Output**: 16kHz (Whisper standard)
- **Rationale**: Whisper models are trained on 16kHz audio

### 4. Amplitude Normalization
- **Purpose**: Prevent clipping and ensure consistent volume levels
- **Method**: Normalize to [-1, 1] range: `audio = audio / max(abs(audio))`
- **Rationale**: Prevents distortion, ensures consistent signal levels for training

### 5. Silence Trimming
- **Purpose**: Remove non-speech segments at beginning and end
- **Method**: `librosa.effects.trim()` with 30dB threshold
- **Rationale**: Reduces file size, focuses on speech content, improves training efficiency

### 6. File Saving
- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Location**: `data/processed_audio/`

---

## Text Preprocessing Steps

### 1. Transcript Extraction
- **Input**: JSON files with timestamped segments
- **Method**: Concatenate all segment texts with spaces
- **Output**: Full transcript text

### 2. Unicode Normalization
- **Purpose**: Ensure consistent character representation
- **Method**: NFKC (Normalization Form Compatibility Composition)
- **Rationale**: Handles various Unicode variants of the same character (e.g., different ways to represent accented characters)

### 3. Quote Normalization
- **Purpose**: Standardize quote characters
- **Changes**:
  - `'` → `'` (left single quotation mark → apostrophe)
  - `'` → `'` (right single quotation mark → apostrophe)
  - `"` → `"` (left double quotation mark → quote)
  - `"` → `"` (right double quotation mark → quote)

### 4. Whitespace Normalization
- **Purpose**: Clean up spacing inconsistencies
- **Method**: 
  - Strip leading/trailing whitespace
  - Replace multiple consecutive spaces with single space
- **Rationale**: Consistent spacing improves model training

### 5. Character Filtering
- **Purpose**: Remove invalid characters, keep only relevant content
- **Keeps**:
  - Hindi Devanagari script: Unicode range U+0900 to U+097F
  - Basic Latin: a-z, A-Z, 0-9
  - Common punctuation: `,`, `.`, `-`, `!`, `?`, `'`
- **Removes**: Special characters, symbols, control characters
- **Rationale**: Prevents encoding issues, focuses on relevant content

### 6. Final Cleanup
- Strip any remaining leading/trailing whitespace
- Validate non-empty text
- Save as plain text files in `data/processed_transcript/`

---

## Dataset Statistics

### Input Dataset
- **Format**: CSV file (`ft_data.csv`) with paths to audio and transcript files
- **Audio Format**: Various formats, typically WAV, various sample rates
- **Transcript Format**: JSON files with timestamped segments

### Output Dataset
- **Format**: CSV file (`preprocessed_data.csv`) with paths to processed files
- **Audio Format**: WAV, 16kHz, Mono, Normalized, Trimmed
- **Transcript Format**: Plain text files with normalized Hindi text

### Processing Results
- **Total Files Processed**: [Run preprocessing to see count]
- **Total Audio Duration**: [Run preprocessing to see duration]
- **Average File Duration**: [Calculated from processed files]

---

## Quality Assurance

### Audio Quality Checks
- ✅ All audio files resampled to 16kHz
- ✅ All audio converted to mono
- ✅ Amplitude normalized to [-1, 1]
- ✅ Silence trimmed from edges
- ✅ Files saved in WAV format

### Text Quality Checks
- ✅ Unicode normalization applied
- ✅ Whitespace normalized
- ✅ Only valid Hindi characters retained
- ✅ Empty transcripts filtered out
- ✅ UTF-8 encoding validated

---

## File Structure

### Before Preprocessing
```
data/
├── audio/
│   └── {recording_id}.wav (various formats, sample rates)
├── transcript/
│   └── {recording_id}.json (timestamped segments)
└── ft_data.csv (original metadata)
```

### After Preprocessing
```
data/
├── processed_audio/
│   └── {recording_id}.wav (16kHz, mono, normalized, trimmed)
├── processed_transcript/
│   └── {recording_id}.txt (normalized plain text)
└── preprocessed_data.csv (paths to processed files)
```

---

## Running Preprocessing

### Command
```bash
python src/preprocess.py
```

### What It Does
1. Reads `data/ft_data.csv`
2. For each recording:
   - Processes audio file → saves to `processed_audio/`
   - Processes transcript → saves to `processed_transcript/`
3. Creates `data/preprocessed_data.csv` with metadata

### Output
- Processed audio files
- Processed transcript files
- CSV file with paths and metadata
- Statistics printed to console

---

## Technical Details

### Libraries Used
- **librosa**: Audio loading, processing, resampling
- **soundfile**: Audio file I/O
- **numpy**: Audio array operations
- **unicodedata**: Unicode normalization
- **re**: Regular expressions for text cleaning

### Configuration
- **Target Sample Rate**: 16,000 Hz
- **Silence Trim Threshold**: 30 dB
- **Audio Format**: WAV, 16-bit PCM
- **Text Encoding**: UTF-8

---

## Notes

1. **Deterministic Processing**: All preprocessing steps are deterministic and reproducible
2. **Original Files Preserved**: Original audio and transcript files remain unchanged
3. **Safe Re-running**: Preprocessing can be safely re-run (overwrites processed files)
4. **Error Handling**: Files that fail processing are skipped (logged in output)
5. **Duration Filtering**: Files longer than 30 seconds may be filtered during training (Whisper's context limit)

---

## Next Steps

After preprocessing:
1. Review `preprocessed_data.csv` to verify processed file paths
2. Run fine-tuning: `python src/finetune_whisper.py`
3. Evaluate models: `python src/evaluate.py`

---

## References

- Whisper Model Requirements: 16kHz mono audio
- Unicode Normalization: NFKC standard (RFC 3454)
- Audio Processing: Librosa documentation

