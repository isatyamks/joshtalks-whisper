"""
Evaluation script for Whisper models on FLEURS Hindi test dataset.
Evaluates both baseline Whisper-small and fine-tuned model.
"""

import os
import json
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm
import pandas as pd

# Configuration
BASELINE_MODEL = "openai/whisper-small"
FINE_TUNED_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper-small-hindi')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_CSV = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
RESULTS_JSON = os.path.join(OUTPUT_DIR, 'evaluation_results.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_fleurs_hindi_test():
    """
    Load FLEURS Hindi test dataset.
    
    IMPORTANT: This function will download:
    1. Dataset metadata/index files (small, ~few MB for all languages)
    2. Only Hindi test audio files (~2-3 GB, not the entire 350GB dataset)
    
    The filtering happens BEFORE audio files are downloaded, so only Hindi
    samples will be downloaded, not all 102 languages.
    """
    print("=" * 60)
    print("Loading FLEURS Hindi Test Dataset")
    print("=" * 60)
    print("📥 Download info:")
    print("   - Metadata/index files: Small (~few MB for all languages)")
    print("   - Audio files: Only Hindi test samples (~2-3 GB)")
    print("   - NOT downloading: Other languages (saves ~340+ GB)")
    print("=" * 60)
    
    # FIRST: Try direct language config (most efficient - only downloads Hindi)
    # This works with datasets library versions that support language configs
    try:
        print("\n🎯 Attempting direct load with 'hi_in' config...")
        print("   (This downloads ONLY Hindi data if supported by your datasets version)")
        dataset = load_dataset("google/fleurs", "hi_in", split="test")
        print("✅ Success! Loaded Hindi dataset directly - only Hindi data will be downloaded.")
        
    except Exception as direct_error:
        # Direct config not supported - try streaming mode
        print(f"   Direct config not available: {direct_error}")
        print("\n📦 Trying streaming mode (filters before downloading)...")
        
        try:
            # Use streaming WITHOUT Parquet to filter before downloading
            dataset_stream = load_dataset(
                "google/fleurs", 
                "default", 
                split="test",
                streaming=True  # Streaming mode filters before download
            )
            
            print("✓ Connected to dataset stream")
            print("   Filtering for Hindi samples...")
            
            # Filter for Hindi samples in stream
            def is_hindi_sample(example):
                sample_id = str(example.get('id', ''))
                return 'hi_in' in sample_id
            
            hindi_stream = dataset_stream.filter(is_hindi_sample)
            
            # Convert stream to dataset (only downloads filtered samples)
            from datasets import Dataset
            hindi_samples = []
            
            print("   Downloading only Hindi samples (this may take a few minutes)...")
            for sample in tqdm(hindi_stream, desc="Downloading Hindi samples"):
                hindi_samples.append(sample)
                # Safety limit
                if len(hindi_samples) >= 5000:
                    print(f"   Reached 5000 samples limit")
                    break
            
            if len(hindi_samples) == 0:
                raise ValueError("No Hindi samples found in stream")
            
            dataset = Dataset.from_list(hindi_samples)
            print(f"✅ Loaded {len(dataset)} Hindi samples via streaming")
            
        except Exception as stream_error:
            # Standard streaming failed - try Parquet format with streaming
            print(f"   Standard streaming failed: {stream_error}")
            print("\n📦 Trying Parquet format with streaming...")
            
            try:
                # Try Parquet format with streaming
                dataset_stream = load_dataset(
                    "google/fleurs", 
                    "default", 
                    split="test",
                    revision="convert/parquet",
                    streaming=True
                )
                
                print("✓ Connected to Parquet stream")
                print("   Filtering for Hindi samples...")
                
                def is_hindi_sample(example):
                    sample_id = str(example.get('id', ''))
                    return 'hi_in' in sample_id
                
                hindi_stream = dataset_stream.filter(is_hindi_sample)
                
                # Convert streaming dataset to regular dataset
                # Note: Audio decoding will happen during iteration, which requires torchcodec
                from datasets import Dataset
                print("   Converting stream to dataset (downloading Hindi Parquet files)...")
                print("   ⚠️  Note: Audio decoding requires 'torchcodec' package")
                
                # Collect samples from stream
                # Audio will be decoded during iteration, so torchcodec is needed
                hindi_samples = []
                for sample in tqdm(hindi_stream, desc="Downloading Hindi samples"):
                    hindi_samples.append(sample)
                    if len(hindi_samples) >= 5000:
                        break
                
                if len(hindi_samples) == 0:
                    raise ValueError("No Hindi samples found")
                
                dataset = Dataset.from_list(hindi_samples)
                print(f"✅ Loaded {len(dataset)} Hindi samples via Parquet streaming")
                print("   (Audio files will be decoded when accessed during evaluation)")
                
            except ImportError as parquet_error:
                # Check if it's the torchcodec error
                if 'torchcodec' in str(parquet_error):
                    print(f"\n❌ Parquet streaming requires 'torchcodec' for audio decoding.")
                    print("\n" + "="*60)
                    print("💡 SOLUTION OPTIONS:")
                    print("="*60)
                    print("\nOption 1 (RECOMMENDED - Simplest):")
                    print("   pip install 'datasets==3.6.0'")
                    print("   This enables direct config: load_dataset('google/fleurs', 'hi_in', split='test')")
                    print("   Downloads ONLY Hindi data (~2-3 GB)")
                    print("\nOption 2 (If you want to keep current datasets version):")
                    print("   pip install torchcodec")
                    print("   Then run this script again")
                    print("\nWe recommend Option 1 as it's simpler and more efficient.\n")
                    
                    raise RuntimeError(
                        f"Parquet streaming requires 'torchcodec' for audio decoding.\n\n"
                        f"RECOMMENDED: pip install 'datasets==3.6.0'\n"
                        f"OR: pip install torchcodec\n\n"
                        f"Direct config error: {direct_error}\n"
                        f"Standard streaming error: {stream_error}\n"
                        f"Parquet streaming error: {parquet_error}"
                    ) from parquet_error
                else:
                    raise
            except Exception as parquet_error:
                # All methods failed - MUST downgrade
                print(f"\n❌ Parquet streaming also failed: {parquet_error}")
                print("\n" + "="*60)
                print("⚠️  REQUIRED ACTION: Downgrade datasets library")
                print("="*60)
                print("\nYour datasets library version doesn't support FLEURS dataset scripts.")
                print("You MUST downgrade to use language-specific configs.\n")
                print("📋 Steps:")
                print("   1. Run: pip install 'datasets==3.6.0'")
                print("   2. Run this script again: python src/evaluate.py")
                print("\nThis will enable: load_dataset('google/fleurs', 'hi_in', split='test')")
                print("which downloads ONLY Hindi data (~2-3 GB).\n")
                
                raise RuntimeError(
                    f"All loading methods failed. Your datasets library version doesn't support\n"
                    f"FLEURS dataset scripts. You MUST downgrade:\n\n"
                    f"  pip install 'datasets==3.6.0'\n\n"
                    f"Direct config error: {direct_error}\n"
                    f"Standard streaming error: {stream_error}\n"
                    f"Parquet streaming error: {parquet_error}"
                ) from parquet_error
    
    # Cast audio column - this doesn't download yet, just sets up the format
    # Audio will be downloaded when accessed during evaluation
    print("\n🎵 Step 3: Configuring audio format (16kHz)...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("✓ Audio format configured. Files will download during evaluation.")
    
    print(f"\n✅ Successfully prepared {len(dataset)} Hindi test samples")
    print("=" * 60)
    return dataset


def transcribe_batch(model, processor, audio_arrays, sampling_rates, device, batch_size=8):
    """
    Transcribe a batch of audio files.
    """
    transcriptions = []
    
    for i in tqdm(range(0, len(audio_arrays), batch_size), desc="Transcribing"):
        batch_audio = audio_arrays[i:i+batch_size]
        batch_sr = sampling_rates[i:i+batch_size]
        
        # Process audio
        inputs = processor.feature_extractor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                language="hi",
                task="transcribe"
            )
        
        # Decode
        batch_transcriptions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        transcriptions.extend(batch_transcriptions)
    
    return transcriptions


def evaluate_model(model_path, model_name, dataset, device):
    """
    Evaluate a Whisper model on the FLEURS Hindi test set.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Load model and processor
    print(f"Loading model from: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Prepare data
    print("Preparing test data...")
    audio_arrays = [item["audio"]["array"] for item in dataset]
    sampling_rates = [item["audio"]["sampling_rate"] for item in dataset]
    references = [item["transcription"] for item in dataset]
    
    # Transcribe
    print("Transcribing audio files...")
    predictions = transcribe_batch(model, processor, audio_arrays, sampling_rates, device)
    
    # Compute metrics
    print("Computing metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    results = {
        "model": model_name,
        "model_path": model_path,
        "num_samples": len(dataset),
        "wer": wer,
        "cer": cer,
        "predictions": predictions[:10],  # Store first 10 for inspection
        "references": references[:10]
    }
    
    print(f"\nResults for {model_name}:")
    print(f"  WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.2f}%)")
    
    return results


def main():
    """
    Main evaluation pipeline.
    """
    print("=" * 60)
    print("Whisper Model Evaluation on FLEURS Hindi Test Set")
    print("=" * 60)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = load_fleurs_hindi_test()
    
    # Evaluate baseline model
    baseline_results = evaluate_model(
        BASELINE_MODEL,
        "Whisper-small (Baseline)",
        test_dataset,
        device
    )
    
    # Evaluate fine-tuned model (if it exists)
    fine_tuned_results = None
    if os.path.exists(FINE_TUNED_MODEL) and os.path.exists(os.path.join(FINE_TUNED_MODEL, "config.json")):
        fine_tuned_results = evaluate_model(
            FINE_TUNED_MODEL,
            "Whisper-small (Fine-tuned)",
            test_dataset,
            device
        )
    else:
        print(f"\n⚠ Fine-tuned model not found at: {FINE_TUNED_MODEL}")
        print("  Skipping fine-tuned model evaluation.")
    
    # Compile results
    all_results = {
        "baseline": baseline_results,
        "fine_tuned": fine_tuned_results
    }
    
    # Save results as JSON
    print(f"\nSaving results to: {RESULTS_JSON}")
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Create summary table
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    summary_data = []
    summary_data.append({
        "Model": "Whisper-small (Baseline)",
        "WER (%)": f"{baseline_results['wer']*100:.2f}",
        "CER (%)": f"{baseline_results['cer']*100:.2f}",
        "Num Samples": baseline_results['num_samples']
    })
    
    if fine_tuned_results:
        summary_data.append({
            "Model": "Whisper-small (Fine-tuned)",
            "WER (%)": f"{fine_tuned_results['wer']*100:.2f}",
            "CER (%)": f"{fine_tuned_results['cer']*100:.2f}",
            "Num Samples": fine_tuned_results['num_samples']
        })
        
        # Calculate improvement
        wer_improvement = ((baseline_results['wer'] - fine_tuned_results['wer']) / baseline_results['wer']) * 100
        cer_improvement = ((baseline_results['cer'] - fine_tuned_results['cer']) / baseline_results['cer']) * 100
        
        
        summary_data.append({
            "Model": "Improvement",
            "WER (%)": f"{wer_improvement:+.2f}%",
            "CER (%)": f"{cer_improvement:+.2f}%",
            "Num Samples": "-"
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ Results saved to: {RESULTS_CSV}")
    print(f"✓ Detailed results saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()

