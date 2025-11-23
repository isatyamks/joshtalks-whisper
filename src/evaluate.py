import os
import json
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from tqdm import tqdm
import pandas as pd

import evaluate as evaluate_lib

BASELINE_MODEL = "openai/whisper-small"
FINE_TUNED_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper-small-hindi')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_CSV = os.path.join(OUTPUT_DIR, 'FT Result.csv')
RESULTS_JSON = os.path.join(OUTPUT_DIR, 'FT Result.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_fleurs_hindi_test():
    dataset = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def transcribe_batch(model, processor, audio_arrays, device, batch_size=8):
    transcriptions = []
    for i in tqdm(range(0, len(audio_arrays), batch_size), desc="Transcribing"):
        batch_audio = audio_arrays[i:i+batch_size]
        inputs = processor.feature_extractor(batch_audio, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features, language="hi", task="transcribe")
        batch_transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        transcriptions.extend(batch_transcriptions)
    return transcriptions

def evaluate_model(model_path, model_name, dataset, device):
    processor = WhisperProcessor.from_pretrained(model_path, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    audio_arrays = [item["audio"]["array"] for item in dataset]
    references = [item["transcription"] for item in dataset]
    predictions = transcribe_batch(model, processor, audio_arrays, device)

    wer_metric = evaluate_lib.load("wer")
    cer_metric = evaluate_lib.load("cer")

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    results = {
        "model": model_name,
        "model_path": model_path,
        "num_samples": len(dataset),
        "wer": wer,
        "cer": cer,
        "predictions": predictions[:10],
        "references": references[:10]
    }

    print(f"\nResults for {model_name}:")
    print(f"  WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.2f}%)")

    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = load_fleurs_hindi_test()

    baseline_results = evaluate_model(
        BASELINE_MODEL,
        "Whisper-small (Baseline)",
        test_dataset,
        device
    )

    fine_tuned_results = None
    if os.path.exists(FINE_TUNED_MODEL) and os.path.exists(os.path.join(FINE_TUNED_MODEL, "config.json")):
        fine_tuned_results = evaluate_model(
            FINE_TUNED_MODEL,
            "Whisper-small (Fine-tuned)",
            test_dataset,
            device
        )
    else:
        print(f"Fine-tuned model not found at: {FINE_TUNED_MODEL}")

    all_results = {
        "baseline": baseline_results,
        "fine_tuned": fine_tuned_results
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

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

        wer_improvement = ((baseline_results['wer'] - fine_tuned_results['wer']) / baseline_results['wer']) * 100
        cer_improvement = ((baseline_results['cer'] - fine_tuned_results['cer']) / baseline_results['cer']) * 100

        summary_data.append({
            "Model": "Improvement",
            "WER (%)": f"{wer_improvement:+.2f}%",
            "CER (%)": f"{cer_improvement:+.2f}%",
            "Num Samples": "-"
        })

    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS - FLEURS Hindi Test Set")
    print("="*80)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*80)
    
    print("\nMarkdown Table Format:")
    print("-"*80)
    try:
        print(df.to_markdown(index=False))
    except AttributeError:
        markdown_table = "| " + " | ".join(df.columns) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
        for _, row in df.iterrows():
            markdown_table += "| " + " | ".join(str(val) for val in row) + " |\n"
        print(markdown_table)
    print("-"*80)
    
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to: {RESULTS_CSV}")
    print(f"Detailed results saved to: {RESULTS_JSON}")

if __name__ == "__main__":
    main()
