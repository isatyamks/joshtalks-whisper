"""
Fine-tuning script for Whisper-small model on Hindi ASR dataset.
Uses Hugging Face Transformers and Datasets libraries.
"""

import os
import json
import csv
import torch
import librosa
from datasets import Dataset, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PREPROCESSED_CSV = os.path.join(DATA_DIR, 'preprocessed_data.csv')
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper-small-hindi')
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
MAX_DURATION_IN_SECONDS = 30.0  # Whisper's max context length
MAX_LABEL_LENGTH = 448  # Whisper's maximum label sequence length


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper fine-tuning.
    Pads audio inputs and labels to the same length.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad inputs - ensure we get fresh tensors
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Ensure labels are on the same device as input_features
        if isinstance(batch["input_features"], torch.Tensor):
            labels = labels.to(batch["input_features"].device)
        
        # If decoder_input_ids are present, replace padding with pad_token_id
        if (labels == self.processor.tokenizer.pad_token_id).all().item():
            labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        
        batch["labels"] = labels
        
        return batch


def prepare_dataset(csv_path: str):
    """
    Load and prepare dataset from preprocessed CSV.
    """
    print("Loading dataset from CSV...")
    print(f"CSV path: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Preprocessed CSV not found at: {csv_path}\n"
            f"Please run preprocessing first: python src/preprocess.py"
        )
    
    # Read CSV
    records = []
    total_rows = 0
    filtered_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            # Check if required columns exist
            if 'audio_path' not in row or 'transcript' not in row or 'processed_duration' not in row:
                print(f"Warning: Missing required columns in CSV. Found: {list(row.keys())}")
                continue
            
            # Check if audio file exists
            audio_path = row['audio_path']
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            # Note: We don't filter by duration here since Whisper can handle longer audio
            # The model will automatically chunk long audio during inference
            # For training, we'll let the data collator handle batching
            records.append({
                'audio': audio_path,
                'text': row['transcript']
            })
    
    print(f"Total rows in CSV: {total_rows}")
    print(f"Successfully loaded: {len(records)} samples")
    
    if len(records) == 0:
        raise ValueError(
            f"No valid samples found in CSV!\n"
            f"Please check:\n"
            f"1. CSV file exists and has data\n"
            f"2. Audio files exist at specified paths\n"
            f"3. CSV has required columns: audio_path, transcript, processed_duration"
        )
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(records)
    
    # Note: We don't use Audio column type to avoid torchcodec dependency
    # We'll load audio directly in prepare_dataset_batch using librosa
    
    return dataset


def prepare_dataset_batch(example, processor, max_label_length=448):
    """
    Prepare example for training:
    1. Load audio file using librosa
    2. Extract audio features
    3. Tokenize text labels (truncate if too long)
    """
    # Load audio file directly using librosa
    audio_path = example["audio"]
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Extract features using processor
    example["input_features"] = processor.feature_extractor(
        audio_array, 
        sampling_rate=16000
    ).input_features[0]
    
    # Tokenize text
    tokenized = processor.tokenizer(
        example["text"],
        max_length=max_label_length,
        truncation=True,
        padding=False
    )
    example["labels"] = tokenized.input_ids
    
    # Skip if labels are still too long (safety check)
    if len(example["labels"]) > max_label_length:
        # Truncate to max length
        example["labels"] = example["labels"][:max_label_length]
    
    return example


def compute_metrics(pred):
    """
    Compute WER (Word Error Rate) metric.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


def main():
    """
    Main fine-tuning pipeline.
    """
    print("=" * 60)
    print("Whisper-small Fine-tuning for Hindi ASR")
    print("=" * 60)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load processor and model
    print(f"\nLoading model: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Set language and task tokens
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    
    # Move model to device
    model = model.to(device)
    
    # Load and prepare dataset
    dataset = prepare_dataset(PREPROCESSED_CSV)
    
    # Prepare dataset batches
    print("\nPreparing dataset batches...")
    print("Note: Audio files longer than ~30 seconds may be truncated")
    
    # Filter function to check if sample will be valid
    def is_valid_sample(example):
        """Check if text is not too long (rough estimate)"""
        # Rough estimate: ~10 characters per token for Hindi
        # So 448 tokens ≈ 4500 characters max
        return len(example["text"]) < 4500
    
    # Filter out samples that are likely too long
    initial_count = len(dataset)
    dataset = dataset.filter(is_valid_sample)
    filtered_count = initial_count - len(dataset)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} samples that are too long")
    
    dataset = dataset.map(
        lambda x: prepare_dataset_batch(x, processor, max_label_length=MAX_LABEL_LENGTH),
        remove_columns=dataset.column_names,
        desc="Processing audio and text"
    )
    
    # Additional filter after tokenization to ensure labels are within limit
    def has_valid_labels(example):
        return len(example["labels"]) <= MAX_LABEL_LENGTH
    
    initial_count = len(dataset)
    dataset = dataset.filter(has_valid_labels)
    final_filtered = initial_count - len(dataset)
    if final_filtered > 0:
        print(f"Filtered out {final_filtered} additional samples with labels exceeding max length")
    
    # Split into train and validation (90/10)
    if len(dataset) > 1:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        # If only one sample, use it for both (not ideal but prevents error)
        print("Warning: Only one sample found. Using it for both train and validation.")
        train_dataset = dataset
        eval_dataset = dataset
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("No training samples available!")
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=False,  # Disable to avoid graph issues
        fp16=True if device == "cuda" else False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        save_total_limit=3,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,  # Use processing_class instead of tokenizer
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

