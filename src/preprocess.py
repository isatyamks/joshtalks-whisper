import os
import json
import csv
import librosa
import soundfile as sf
import numpy as np
import unicodedata
import re
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
TRANSCRIPT_DIR = os.path.join(DATA_DIR, 'transcript')
PROCESSED_AUDIO_DIR = os.path.join(DATA_DIR, 'processed_audio')
PROCESSED_TRANSCRIPT_DIR = os.path.join(DATA_DIR, 'processed_transcript')
CSV_PATH = os.path.join(DATA_DIR, 'ft_data.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'preprocessed_data.csv')

TARGET_SR = 16000
TRIM_TOP_DB = 30

os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)
os.makedirs(PROCESSED_TRANSCRIPT_DIR, exist_ok=True)

def normalize_text(text):
    """
    Normalize Hindi text for ASR training.
    """
    text = unicodedata.normalize("NFKC", text).strip()
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F',.\-!?]", " ", text)
    return text.strip()

def extract_full_transcript(transcript_path):
    with open(transcript_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    full_text = " ".join([segment.get('text', '') for segment in segments])
    return normalize_text(full_text)

def process_audio(audio_path, output_path):
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    sf.write(output_path, audio_trimmed, TARGET_SR, format='WAV')
    duration = len(audio_trimmed) / TARGET_SR
    return duration

def main():
    processed_records = []
    with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing files"):
            recording_id = row['recording_id']

            audio_path = os.path.join(AUDIO_DIR, f"{recording_id}.wav")
            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{recording_id}.json")

            processed_audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"{recording_id}.wav")
            processed_transcript_path = os.path.join(PROCESSED_TRANSCRIPT_DIR, f"{recording_id}.txt")

            if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
                continue

            duration = process_audio(audio_path, processed_audio_path)
            transcript_text = extract_full_transcript(transcript_path)
            if not transcript_text or len(transcript_text.strip()) == 0:
                continue

            with open(processed_transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)

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

    if processed_records:
        fieldnames = ['recording_id', 'user_id', 'language', 'original_duration',
                     'processed_duration', 'audio_path', 'transcript_path', 'transcript']

        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_records)

        print(f"Processed {len(processed_records)} files")
        print(f"Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
