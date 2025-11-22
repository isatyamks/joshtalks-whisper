import os
import csv
import requests
from urllib.parse import urlparse

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
TRANSCRIPT_DIR = os.path.join(DATA_DIR, 'transcript')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
CSV_PATH = os.path.join(DATA_DIR, 'ft_data.csv')

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def get_id_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path)
    return filename.split('_')[0] if '_' in filename else filename.split('.')[0]

def download_file(url, dest_path):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Audio
            audio_url = row['rec_url_gcp']
            audio_id = get_id_from_url(audio_url)
            audio_ext = os.path.splitext(audio_url)[1]
            audio_path = os.path.join(AUDIO_DIR, f"{audio_id}{audio_ext}")
            download_file(audio_url, audio_path)

            # Transcript
            transcript_url = row['transcription_url_gcp']
            transcript_id = get_id_from_url(transcript_url)
            transcript_ext = os.path.splitext(transcript_url)[1]
            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{transcript_id}{transcript_ext}")
            download_file(transcript_url, transcript_path)

            # Metadata
            metadata_url = row['metadata_url_gcp']
            metadata_id = get_id_from_url(metadata_url)
            metadata_ext = os.path.splitext(metadata_url)[1]
            metadata_path = os.path.join(METADATA_DIR, f"{metadata_id}{metadata_ext}")
            download_file(metadata_url, metadata_path)


            

if __name__ == "__main__":
    main()
