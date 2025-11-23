"""
Fine-tuning script for Whisper-small model on Hindi ASR dataset.
Uses Hugging Face Transformers and Datasets libraries.
"""

import os
import csv
import torch
import librosa
from datasets import Dataset
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

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PREPROCESSED_CSV = os.path.join(DATA_DIR, 'preprocessed_data.csv')
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper-small-hindi')
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
MAX_LABEL_LENGTH = 448

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper fine-tuning.
    Pads audio inputs and labels to the same length.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if isinstance(batch["input_features"], torch.Tensor):
            labels = labels.to(batch["input_features"].device)
        if (labels == self.processor.tokenizer.pad_token_id).all().item():
            labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch

def prepare_dataset(csv_path: str):
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'audio_path' not in row or 'transcript' not in row:
                continue
            if not os.path.exists(row['audio_path']):
                continue
            records.append({
                'audio': row['audio_path'],
                'text': row['transcript']
            })

    if len(records) == 0:
        raise ValueError("No valid samples found in CSV")
    return Dataset.from_list(records)

def prepare_dataset_batch(example, processor, max_label_length=448):
    audio_array, _ = librosa.load(example["audio"], sr=16000, mono=True)
    example["input_features"] = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
    tokenized = processor.tokenizer(example["text"], max_length=max_label_length, truncation=True, padding=False)
    example["labels"] = tokenized.input_ids[:max_label_length]
    return example

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    model = model.to(device)
    dataset = prepare_dataset(PREPROCESSED_CSV)

    dataset = dataset.filter(lambda x: len(x["text"]) < 4500)

    dataset = dataset.map(
        lambda x: prepare_dataset_batch(x, processor, max_label_length=MAX_LABEL_LENGTH),
        remove_columns=dataset.column_names,
        desc="Processing audio and text"
    )

    dataset = dataset.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)

    if len(dataset) > 1:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = dataset

    if len(train_dataset) == 0:
        raise ValueError("No training samples available!")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=False,
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

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
