# Preprocessing Pipeline: Technical Documentation

## Overview

This document provides a comprehensive technical analysis of the preprocessing pipeline implemented in `src/preprocess.py`. The preprocessing stage is critical for preparing Hindi ASR data for fine-tuning the Whisper-small model, ensuring data consistency, quality, and compatibility with the model's requirements.

## Architecture

The preprocessing pipeline consists of two parallel processing streams: audio preprocessing and text preprocessing. These streams operate independently but are synchronized through the main processing loop that processes paired audio-transcript files.

### Data Flow

```
Raw Audio Files (various formats, sample rates)
    ↓
Audio Preprocessing Pipeline
    ↓
Processed Audio (16kHz, mono, normalized)
    ↓
CSV Metadata Generation
    ↓
preprocessed_data.csv

Raw Transcript Files (JSON with segments)
    ↓
Text Preprocessing Pipeline
    ↓
Processed Transcript (normalized plain text)
    ↓
CSV Metadata Generation
    ↓
preprocessed_data.csv
```

## Audio Preprocessing

### Technical Rationale

Whisper models are trained on 16kHz mono audio. The audio preprocessing pipeline ensures that all input audio conforms to this specification. Non-conformant audio can lead to degraded model performance, longer training times, and inconsistent feature extraction.

### Step-by-Step Analysis

#### Step 1: Audio Loading

**Implementation:**
```python
audio, sr = librosa.load(audio_path, sr=None, mono=False)
```

**Technical Details:**
- Uses Librosa's audio loading function which supports multiple formats (WAV, MP3, FLAC, etc.)
- Setting `sr=None` preserves the original sample rate, allowing us to handle files with different rates
- Setting `mono=False` preserves channel information, enabling explicit stereo-to-mono conversion

**Why This Approach:**
Explicit preservation of original properties allows for controlled transformation. Loading at the original sample rate before resampling avoids unnecessary intermediate conversions that could introduce artifacts.

**Potential Issues and Mitigation:**
- Corrupted files: Librosa handles most corruptions gracefully, returning errors that can be caught
- Unsupported formats: Librosa supports most common formats, but exotic formats may require conversion
- Memory constraints: For very large files, streaming approaches would be needed, but typical ASR datasets have reasonable file sizes

#### Step 2: Mono Conversion

**Implementation:**
```python
if len(audio.shape) > 1:
    audio = librosa.to_mono(audio)
```

**Technical Details:**
- Checks if audio has multiple channels by examining the shape of the audio array
- `librosa.to_mono()` averages stereo channels: `mono = (left + right) / 2`
- This is a simple averaging method, which is standard for speech signals

**Why Mono:**
- Whisper models expect single-channel audio
- Reduces data dimensionality (2D array to 1D)
- Consistent with training data format
- Saves storage space

**Trade-offs:**
Simple averaging works well for speech because most speech signals are centered in the stereo field. For cases where channels contain different information (e.g., one channel has noise), more sophisticated methods like channel selection could be used, but averaging is the standard approach.

#### Step 3: Resampling to 16kHz

**Implementation:**
```python
if sr != TARGET_SR:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
```

**Technical Details:**
- TARGET_SR constant set to 16000 Hz
- Librosa uses sinc interpolation for resampling, which is high-quality
- Only resamples if sample rate differs (avoids unnecessary processing)

**Why 16kHz:**
- Whisper models were trained on 16kHz audio
- This sample rate is sufficient for human speech (Nyquist frequency of 8kHz covers speech frequencies up to 4kHz, with harmonics extending higher)
- Balancing quality and computational efficiency
- Standard in speech recognition systems

**Resampling Quality:**
Librosa's resampler uses a high-quality sinc interpolation method. This is computationally more expensive than linear interpolation but provides better frequency response. For speech signals, this quality is important to preserve formant structure.

**Potential Issues:**
- Aliasing: Prevented by Librosa's anti-aliasing filters built into the resampler
- Computational cost: Resampling is CPU-intensive but necessary and unavoidable
- Quality loss: Some high-frequency information is lost when downsampling from 44.1kHz or 48kHz, but this is acceptable for speech

#### Step 4: Amplitude Normalization

**Implementation:**
```python
max_abs = np.max(np.abs(audio))
if max_abs > 0:
    audio = audio / max_abs
```

**Technical Details:**
- Computes the maximum absolute value across the entire audio signal
- Divides all samples by this maximum value, scaling to [-1, 1] range
- Only normalizes if max_abs > 0 to avoid division by zero (handles silent files)

**Why Normalize:**
- Prevents clipping in feature extraction pipelines
- Ensures consistent signal amplitude across files with different recording levels
- Whisper's feature extractor expects normalized audio
- Reduces the impact of recording gain variations

**Normalization Type:**
This is peak normalization (also called max normalization). Alternative approaches include:
- RMS normalization: Normalizes based on root mean square, more robust to outliers
- Loudness normalization: EBU R128 standard, perceptually motivated

Peak normalization is chosen for simplicity and effectiveness with clean recordings. RMS normalization would be better for noisy data.

**Edge Cases:**
- Silent files: Check prevents division by zero
- Files already normalized: Division by 1.0 has no effect, operation is idempotent
- Files with DC offset: Normalization preserves offset but scales it proportionally

#### Step 5: Silence Trimming

**Implementation:**
```python
audio_trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
```

**Technical Details:**
- TRIM_TOP_DB constant set to 30 dB
- Librosa's trim function uses energy-based detection
- Removes leading and trailing silence
- Returns trimmed audio and the frame indices where trimming occurred (discarded here)

**Why Trim:**
- Removes non-speech segments that don't contribute to learning
- Reduces file sizes and training time
- Focuses model attention on actual speech content
- Removes recording artifacts like microphone handling noise

**Threshold Selection (30 dB):**
- 30 dB represents roughly a 1000x difference in power (10^(30/10) ≈ 1000)
- This threshold effectively captures silence and background noise
- Conservative threshold: lower values (20-25 dB) would trim more aggressively
- Higher values (35-40 dB) would trim less, preserving more context

**Trade-offs:**
More aggressive trimming (lower threshold) removes more silence but risks removing quiet speech. Less aggressive trimming (higher threshold) preserves more context but includes more non-speech. 30 dB is a balanced choice validated in speech recognition literature.

**Limitations:**
- Only trims beginning and end, not mid-sentence pauses
- Energy-based detection may fail on very noisy recordings
- May remove intentional pauses in speech (usually acceptable)

#### Step 6: File Saving

**Implementation:**
```python
sf.write(output_path, audio_trimmed, TARGET_SR, format='WAV')
duration = len(audio_trimmed) / TARGET_SR
```

**Technical Details:**
- Uses SoundFile library for writing (faster and more reliable than Librosa for writing)
- WAV format: uncompressed, lossless
- 16-bit PCM encoding (SoundFile default)
- Duration calculated from array length and sample rate

**File Format Choice:**
- WAV: Lossless, widely supported, no compression artifacts
- 16-bit: Sufficient dynamic range (96 dB) for speech
- PCM: Standard encoding, no compression overhead

**Storage Considerations:**
For a 10-hour dataset at 16kHz mono 16-bit:
- Storage: 10 hours × 3600 seconds/hour × 16000 samples/second × 2 bytes/sample ≈ 1.15 GB
- This is reasonable for modern storage systems

**Duration Calculation:**
Returns duration in seconds for metadata. Useful for dataset statistics and filtering long files during training.

## Text Preprocessing

### Technical Rationale

Text preprocessing ensures transcript consistency and compatibility with the tokenizer. Inconsistent text formats can lead to tokenization issues, vocabulary bloat, and reduced model performance. For Hindi text, proper Unicode normalization and character filtering are particularly important.

### Step-by-Step Analysis

#### Step 1: Transcript Extraction

**Implementation:**
```python
with open(transcript_path, 'r', encoding='utf-8') as f:
    segments = json.load(f)
full_text = " ".join([segment.get('text', '') for segment in segments])
```

**Technical Details:**
- Reads JSON file with UTF-8 encoding (supports Hindi Unicode characters)
- Assumes JSON structure contains an array of segment objects
- Each segment has a 'text' field containing transcript text
- Segments are joined with single spaces

**Data Structure Assumption:**
The code assumes a specific JSON structure:
```json
[
    {"text": "first segment", "start": 0.0, "end": 2.5},
    {"text": "second segment", "start": 2.5, "end": 5.0}
]
```

**Error Handling:**
- Uses `.get('text', '')` to handle missing 'text' fields gracefully
- Empty segments result in empty strings, which are harmless when joined
- No explicit error handling for malformed JSON (would raise exception)

**Space Joining:**
Segments are joined with single spaces. This assumes segments don't have trailing/leading spaces that would create double spaces. This is typically the case, but normalization handles it anyway.

#### Step 2: Unicode Normalization

**Implementation:**
```python
text = unicodedata.normalize("NFKC", text).strip()
```

**Technical Details:**
- NFKC: Normalization Form Compatibility Composition
- Combines compatibility decomposition and canonical composition
- Handles multiple Unicode representations of the same character

**Why NFKC:**
For Hindi text, Unicode normalization is crucial because:
- Diacritics can be represented as combining characters or precomposed characters
- Various Unicode standards may encode the same visual character differently
- NFKC provides canonical forms while preserving compatibility

**Example:**
The Hindi character "क" can be represented as:
- U+0915 (क) - Devanagari letter KA
- Or decomposed forms

NFKC ensures consistent representation.

**Trade-offs:**
- NFKC: More aggressive, may change some characters
- NFC: Less aggressive, preserves more variants
- NFKD/NFD: Decomposed forms, not suitable for our use case

NFKC is chosen for maximum consistency, which is important for vocabulary building.

#### Step 3: Quote Normalization

**Implementation:**
```python
text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
```

**Technical Details:**
- Replaces smart quotes (curly quotes) with standard straight quotes
- Handles both single and double quotes
- Left and right variants are normalized to the same character

**Why Normalize Quotes:**
- Smart quotes can appear in transcripts from word processors or transcription tools
- Tokenizers typically treat smart quotes as separate tokens from straight quotes
- Normalization reduces vocabulary size and improves consistency
- Straight quotes are more standard in plain text

**Limitations:**
This only handles the most common smart quote characters. Other Unicode quote variants (e.g., U+2018, U+2019, U+201C, U+201D) would need additional replacements if present in the data.

#### Step 4: Whitespace Normalization

**Implementation:**
```python
text = re.sub(r"\s+", " ", text)
```

**Technical Details:**
- Regular expression replaces one or more whitespace characters with a single space
- `\s+` matches any whitespace: spaces, tabs, newlines, etc.
- Replaces with single space character

**Why Normalize Whitespace:**
- Different systems may use different whitespace characters (tabs, multiple spaces, etc.)
- Consistent spacing improves tokenization
- Reduces variations in the text representation
- Makes the data cleaner and easier to process

**Edge Cases:**
- Leading/trailing whitespace: Handled by initial `.strip()` call
- Newlines in text: Converted to spaces (typically desired)
- Tabs: Converted to spaces (typically desired)

#### Step 5: Character Filtering

**Implementation:**
```python
text = re.sub(r"[^\w\s\u0900-\u097F',.\-!?]", " ", text)
```

**Technical Details:**
- Regular expression matches characters NOT in the allowed set
- `\w`: Word characters (letters, digits, underscore) in ASCII range
- `\s`: Whitespace characters
- `\u0900-\u097F`: Unicode range for Devanagari script (Hindi)
- Allowed punctuation: `'`, `,`, `.`, `-`, `!`, `?`
- Replaced characters become spaces (not removed)

**Allowed Characters:**
- ASCII word characters (a-z, A-Z, 0-9, _)
- Hindi Devanagari characters (U+0900 to U+097F)
- Basic punctuation marks

**Why Filter:**
- Removes special characters that may cause tokenization issues
- Reduces vocabulary size by eliminating rare characters
- Focuses on linguistic content
- Prevents encoding issues downstream

**Unicode Range:**
U+0900-U+097F covers:
- Devanagari characters (Hindi, Marathi, Sanskrit)
- Diacritical marks
- Vowel signs
- Consonants and numbers

This range is sufficient for Hindi text in Devanagari script.

**Punctuation Choices:**
The allowed punctuation marks are the most common in Hindi transcriptions:
- Apostrophe: For contractions (e.g., "don't")
- Comma: For pauses
- Period: For sentence endings
- Hyphen: For compound words
- Exclamation/Question: For sentence types

Other punctuation is removed, which is typically acceptable for ASR training.

#### Step 6: Final Cleanup

**Implementation:**
```python
return text.strip()
```

**Technical Details:**
- Removes leading and trailing whitespace after all processing
- Ensures clean output with no edge whitespace

**Why Necessary:**
Character filtering may introduce spaces at the beginning or end. This final strip ensures clean output.

## Main Processing Loop

### Architecture

The main function orchestrates the preprocessing pipeline:

1. Iterates through CSV file containing file paths
2. For each recording:
   - Validates file existence
   - Processes audio file
   - Processes transcript file
   - Creates metadata record
3. Writes consolidated CSV with all metadata

### Error Handling Strategy

**File Existence Checks:**
```python
if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
    continue
```

Missing files are silently skipped. This allows the pipeline to continue processing even if some files are missing, which is important for robustness with real-world datasets.

**Empty Transcript Filtering:**
```python
if not transcript_text or len(transcript_text.strip()) == 0:
    continue
```

Empty transcripts are skipped. This handles cases where JSON parsing succeeds but produces empty text (e.g., all segments are empty).

**Error Propagation:**
Errors in audio processing or transcript extraction are not explicitly caught in the main loop. If `process_audio()` or `extract_full_transcript()` raise exceptions, the script will fail. This is acceptable for batch processing where we want to identify problematic files early.

### Metadata Generation

**CSV Structure:**
The output CSV contains:
- `recording_id`: Unique identifier
- `user_id`: Speaker identifier (preserved from input)
- `language`: Language code (preserved from input)
- `original_duration`: Original audio duration from input CSV
- `processed_duration`: Actual processed audio duration (may differ after trimming)
- `audio_path`: Path to processed audio file
- `transcript_path`: Path to processed transcript file
- `transcript`: Full transcript text (for convenience)

**Why Include Transcript in CSV:**
Including the transcript text in the CSV allows for quick inspection without reading individual files. However, for large datasets, this increases CSV size significantly. The transcript could be removed if storage is a concern, but for typical ASR datasets, it's manageable.

## Performance Considerations

### Computational Complexity

**Audio Processing:**
- Loading: O(n) where n is number of samples
- Resampling: O(n log n) for high-quality resampling
- Trimming: O(n) for energy calculation and trimming
- Overall: O(n log n) dominated by resampling

**Text Processing:**
- JSON parsing: O(m) where m is file size
- String operations: O(m) for each operation
- Overall: O(m) linear with file size

### Memory Usage

**Audio Processing:**
- Entire audio file loaded into memory
- For a 1-minute 44.1kHz stereo file: ~10.6 MB
- After processing to 16kHz mono: ~1.9 MB
- Peak memory: Original file size + processed size

**Text Processing:**
- Entire transcript loaded into memory
- Typically < 10 KB per file
- Negligible memory footprint

### Scalability

**Single-Threaded Processing:**
Current implementation processes files sequentially. For large datasets:
- Parallelization could be added using multiprocessing
- Each file processing is independent, making parallelization straightforward
- I/O operations could become bottleneck with many parallel workers

**Storage I/O:**
- Reading and writing files is typically the bottleneck
- SSD storage significantly improves performance
- Network storage (NFS, S3) would be slower but feasible

## Quality Assurance

### Validation Steps

1. **Audio Validation:**
   - Sample rate verification: Processed files should be 16kHz
   - Mono verification: Files should have single channel
   - Amplitude range: Should be within [-1, 1]
   - Duration reasonableness: Should match expected duration

2. **Text Validation:**
   - UTF-8 encoding: All text should be valid UTF-8
   - Character set: Should only contain allowed characters
   - Length reasonableness: Should correlate with audio duration

### Statistics Reporting

The pipeline reports:
- Number of files processed
- Total audio duration in hours
- Output file location

These statistics help validate the preprocessing and provide dataset insights.

## Best Practices

### Reproducibility

- All processing steps are deterministic (no random operations)
- Constants are clearly defined at module level
- Same input always produces same output

### Maintainability

- Constants extracted to module level for easy modification
- Functions are modular and testable independently
- Clear separation of concerns (audio vs text processing)

### Extensibility

The pipeline is easily extensible:
- Additional audio processing steps can be added to `process_audio()`
- Additional text processing steps can be added to `normalize_text()`
- Error handling can be enhanced without changing core logic

## Limitations and Future Improvements

### Current Limitations

1. **Error Handling:**
   - No detailed error logging
   - Failed files are silently skipped without reporting
   - No retry mechanism for transient failures

2. **Validation:**
   - No automatic quality checks on processed files
   - No verification that processed audio matches expected format
   - No transcript validation (e.g., character count sanity checks)

3. **Performance:**
   - Single-threaded processing
   - No progress persistence (can't resume interrupted runs)
   - No incremental processing (always processes all files)

### Potential Improvements

1. **Enhanced Error Handling:**
   - Log failed files to separate error log
   - Report statistics on failures
   - Retry mechanism for network-related failures

2. **Quality Validation:**
   - Automated checks on processed files
   - Statistical validation (duration distributions, etc.)
   - Sample-based manual review workflow

3. **Performance Optimization:**
   - Parallel processing with multiprocessing
   - Incremental processing (skip already-processed files)
   - Progress persistence for resumable runs

4. **Configuration:**
   - Config file for parameters (sample rate, trim threshold, etc.)
   - Command-line arguments for flexibility
   - Environment-specific settings

## Conclusion

The preprocessing pipeline provides a robust, efficient, and maintainable solution for preparing Hindi ASR data for Whisper fine-tuning. The design prioritizes correctness and consistency over performance, which is appropriate for a preprocessing stage that runs once per dataset. The modular architecture allows for easy extension and modification as requirements evolve.

