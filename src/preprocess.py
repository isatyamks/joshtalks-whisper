import os
import json
import csv
import librosa
import soundfile as sf
import numpy as np
import unicodedata
import re
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
TRANSCRIPT_DIR = os.path.join(DATA_DIR, 'transcript')
PROCESSED_AUDIO_DIR = os.path.join(DATA_DIR, 'processed_audio')
PROCESSED_TRANSCRIPT_DIR = os.path.join(DATA_DIR, 'processed_transcript')
CSV_PATH = os.path.join(DATA_DIR, 'ft_data.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'preprocessed_data.csv')

# Audio processing parameters
TARGET_SR = 16000  # Whisper uses 16kHz
TRIM_TOP_DB = 30  # Trim silence threshold

os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
os.makedirs(PROCESSED_TRANSCRIPT_DIR, exist_ok=True)


def normalize_text(text):
    """
    Normalize Hindi text for ASR training.
    - Normalize Unicode characters (NFKC)
    - Remove extra whitespace
    - Normalize quotes
    - Keep only valid characters (Hindi Devanagari + basic punctuation)
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Strip whitespace
    text = text.strip()
    
    # Replace smart quotes
    text = text.replace("'", "'")
    text = text.replace("'", "'")
    text = text.replace(""", '"')
    text = text.replace(""", '"')
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Keep Hindi Devanagari (U+0900 to U+097F), basic Latin, digits, and common punctuation
    text = re.sub(r"[^\w\s\u0900-\u097F',.\-!?]", " ", text)
    
    return text.strip()


def extract_full_transcript(transcript_path):
    """
    Extract full transcript text from JSON file.
    The transcript JSON contains segments with start, end, and text.
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        # Combine all segment texts
        full_text = " ".join([segment.get('text', '') for segment in segments])
        return normalize_text(full_text)
    except Exception as e:
        print(f"Error reading transcript {transcript_path}: {e}")
        return None


def process_audio(audio_path, output_path):
    """
    Process audio file for Whisper training:
    1. Load audio (handles mono/stereo)
    2. Convert to mono if needed
    3. Resample to 16kHz
    4. Normalize amplitude
    5. Trim silence
    6. Save as WAV
    """
    try:
        # Load audio (preserves original sample rate)
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Resample to target sample rate
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        
        # Normalize amplitude to [-1, 1]
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs
        
        # Trim silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
        
        # Save processed audio
        sf.write(output_path, audio_trimmed, TARGET_SR, format='WAV')
        
        duration = len(audio_trimmed) / TARGET_SR
        return True, duration
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return False, 0.0


def main():
    """
    Main preprocessing pipeline:
    1. Read CSV with dataset metadata
    2. For each recording:
       - Process audio file
       - Extract and normalize transcript
       - Save processed files
    3. Create new CSV with paths to processed files
    """
    print("Starting preprocessing pipeline...")
    print(f"Reading data from: {CSV_PATH}")
    
    processed_records = []
    failed_records = []
    
    with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        total_rows = sum(1 for _ in open(CSV_PATH)) - 1  # Exclude header
        
        csvfile.seek(0)
        reader = csv.DictReader(csvfile)
        
        for row in tqdm(reader, total=total_rows, desc="Processing files"):
            recording_id = row['recording_id']
            
            # Paths
            audio_path = os.path.join(AUDIO_DIR, f"{recording_id}.wav")
            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{recording_id}.json")
            
            processed_audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"{recording_id}.wav")
            processed_transcript_path = os.path.join(PROCESSED_TRANSCRIPT_DIR, f"{recording_id}.txt")
            
            # Check if files exist
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                failed_records.append(recording_id)
                continue
            
            if not os.path.exists(transcript_path):
                print(f"Warning: Transcript file not found: {transcript_path}")
                failed_records.append(recording_id)
                continue
            
            # Process audio
            success, duration = process_audio(audio_path, processed_audio_path)
            if not success:
                failed_records.append(recording_id)
                continue
            
            # Process transcript
            transcript_text = extract_full_transcript(transcript_path)
            if transcript_text is None or len(transcript_text.strip()) == 0:
                print(f"Warning: Empty transcript for {recording_id}")
                failed_records.append(recording_id)
                continue
            
            # Save processed transcript
            with open(processed_transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            # Create record for output CSV
            processed_records.append({
                'recording_id': recording_id,
                'user_id': row['user_id'],
                'language': row['language'],
                'original_duration': row['duration'],
                'processed_duration': f"{duration:.2f}",
                'audio_path': processed_audio_path,
                'transcript_path': processed_transcript_path,
                'transcript': transcript_text
            })
    
    # Save processed data CSV
    if processed_records:
        fieldnames = ['recording_id', 'user_id', 'language', 'original_duration', 
                     'processed_duration', 'audio_path', 'transcript_path', 'transcript']
        
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_records)
        
        print(f"\n✓ Successfully processed {len(processed_records)} files")
        print(f"✓ Processed data saved to: {OUTPUT_CSV}")
        
        # Calculate total duration
        total_duration = sum(float(r['processed_duration']) for r in processed_records)
        print(f"✓ Total processed audio duration: {total_duration/3600:.2f} hours")
    else:
        print("\n✗ No files were successfully processed!")
    
    if failed_records:
        print(f"\n⚠ Failed to process {len(failed_records)} files:")
        for rec_id in failed_records[:10]:  # Show first 10
            print(f"  - {rec_id}")
        if len(failed_records) > 10:
            print(f"  ... and {len(failed_records) - 10} more")


if __name__ == "__main__":
    main()

