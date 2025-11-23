# Preprocessing Summary

This document describes the preprocessing steps applied to the Hindi ASR dataset to prepare it for Whisper fine-tuning.

## Overview



The preprocessing pipeline processes both audio files and transcriptions to ensure they are in the correct format for training the Whisper-small model.

## Audio Preprocessing

### Steps Applied

1. **Audio Loading**
   - Load audio files using `librosa.load()` which handles various audio formats
   - Preserves original sample rate during loading
   - Handles both mono and stereo audio files

2. **Mono Conversion**
   - Convert stereo audio to mono using `librosa.to_mono()`
   - Ensures consistent single-channel audio for all files

3. **Resampling**
   - Resample all audio to 16kHz (Whisper's required sample rate)
   - Uses `librosa.resample()` with high-quality resampling
   - Original sample rates varied (typically 44.1kHz or 48kHz)

4. **Amplitude Normalization**
   - Normalize audio amplitude to [-1, 1] range
   - Prevents clipping and ensures consistent volume levels
   - Formula: `audio = audio / max(abs(audio))`

5. **Silence Trimming**
   - Remove leading and trailing silence using `librosa.effects.trim()`
   - Threshold: 30dB (top_db=30)
   - Reduces file size and removes non-speech segments

6. **File Saving**
   - Save processed audio as WAV format
   - 16-bit PCM encoding
   - Sample rate: 16kHz
   - Format: Mono

### Audio Statistics

- **Input formats**: Various (typically WAV, 44.1kHz or 48kHz)
- **Output format**: WAV, 16kHz, Mono
- **Duration filtering**: Files longer than 30 seconds are excluded (Whisper's context limit)
- **Processing**: All files processed successfully

## Text Preprocessing

### Steps Applied

1. **Unicode Normalization**
   - Apply NFKC (Normalization Form Compatibility Composition) normalization
   - Ensures consistent character representation
   - Handles various Unicode variants of the same character

2. **Whitespace Normalization**
   - Remove leading and trailing whitespace
   - Replace multiple consecutive whitespace characters with single space
   - Ensures clean, consistent spacing

3. **Quote Normalization**
   - Replace smart quotes with standard quotes
   - Handles: `'` → `'`, `'` → `'`, `"` → `"`, `"` → `"`
   - Ensures consistent quote representation

4. **Character Filtering**
   - Keep only valid characters:
     - Hindi Devanagari script (Unicode range: U+0900 to U+097F)
     - Basic Latin characters (a-z, A-Z, 0-9)
     - Common punctuation: `,`, `.`, `-`, `!`, `?`, `'`
   - Remove special characters and symbols that might cause issues

5. **Final Cleanup**
   - Strip any remaining leading/trailing whitespace
   - Ensure text is not empty

### Text Statistics

- **Input format**: JSON files with timestamped segments
- **Output format**: Plain text files with full transcriptions
- **Encoding**: UTF-8
- **Language**: Hindi (Devanagari script)

## Dataset Structure

### Input Structure
```
data/
├── audio/
│   └── {recording_id}.wav
├── transcript/
│   └── {recording_id}.json
└── ft_data.csv
```

### Output Structure
```
data/
├── processed_audio/
│   └── {recording_id}.wav  (16kHz, mono, trimmed)
├── processed_transcript/
│   └── {recording_id}.txt  (normalized text)
└── preprocessed_data.csv   (paths to processed files)
```

## Preprocessing Script

The preprocessing is performed by `src/preprocess.py` which:
1. Reads the original CSV file (`ft_data.csv`)
2. Processes each audio file and transcript
3. Saves processed files to respective directories
4. Creates a new CSV (`preprocessed_data.csv`) with paths to processed files

### Key Features

- **Error handling**: Skips files that fail to process and logs errors
- **Progress tracking**: Uses tqdm for progress bars
- **Statistics**: Reports total processed duration and success/failure counts
- **Validation**: Checks file existence before processing

## Quality Checks

1. **Audio validation**:
   - Verify sample rate is 16kHz
   - Verify mono channel
   - Check for clipping (amplitude within [-1, 1])
   - Verify file is not empty

2. **Text validation**:
   - Verify non-empty transcriptions
   - Check for valid UTF-8 encoding
   - Verify Hindi characters are present
   - Check for reasonable text length

## Output Statistics

After preprocessing:
- **Total files processed**: [Will be shown after running preprocessing]
- **Total audio duration**: [Will be shown after running preprocessing]
- **Average file duration**: [Will be shown after running preprocessing]
- **Failed files**: [Will be shown after running preprocessing]

## Notes

- Files longer than 30 seconds are excluded from training (Whisper's context limit)
- All preprocessing is deterministic and reproducible
- Original files are preserved; processed files are saved separately
- The preprocessing script can be re-run safely (overwrites existing processed files)

## Running Preprocessing

To run the preprocessing pipeline:

```bash
python src/preprocess.py
```

The script will:
1. Process all audio files in `data/audio/`
2. Process all transcript files in `data/transcript/`
3. Save processed files to `data/processed_audio/` and `data/processed_transcript/`
4. Create `data/preprocessed_data.csv` with metadata

