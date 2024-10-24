import os
import re
import torch
from speechbrain.inference import EncoderClassifier  # Make sure to import the correct module
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# Load the dataset
dataset = load_from_disk("gujarati_tts_dataset")  # Ensure your dataset path is correct
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
tokenizer = processor.tokenizer

# Function to inspect a few samples in the dataset
print("Dataset samples:", dataset[2:5])

# Prepare data loader
MAX_AUDIO_LENGTH = 16000

def extract_all_chars(batch):
    # For each text in the batch, create a unique vocabulary entry
    vocab = []
    for text in batch["text"]:
        vocab.extend(set(text))  # Extract unique characters from each text entry
    return {"vocab": [list(set(vocab))]}  # Return a list of unique characters

# Map the vocabulary extraction across the dataset
vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names
)

# Ensure the output is correctly formatted
if "vocab" in vocabs.column_names:
    vocabs = vocabs.flatten_indices()  # Ensure it's flat
    unique_vocab = set()
    for v in vocabs["vocab"]:
        unique_vocab.update(v)  # Combine all vocab lists
    print(f"Unique vocabulary size: {len(unique_vocab)}")  # Print size of unique vocabulary

# Load speaker model for embeddings
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
device = "cuda" if torch.cuda.is_available() else "cpu"

speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

# Define a directory to save the speaker embeddings
embeddings_dir = "speaker_embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

def create_speaker_embedding(waveform, index):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        
        # Save the embeddings to a file
        embedding_path = os.path.join(embeddings_dir, f"embedding_{index}.npy")
        np.save(embedding_path, speaker_embeddings.squeeze().cpu().numpy())
        
        return speaker_embeddings.squeeze().cpu().numpy()

def prepare_dataset(example, index):
    audio = example["audio"]
    example = processor(
        text=example["text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    example["labels"] = example["labels"][0]  # Ensure the label shape is correct
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"], index)
    return example

# Check if the dataset is not empty before processing
if len(dataset) > 0:
    # Apply the prepare_dataset function to the entire dataset with index
    dataset = dataset.map(lambda x, idx: prepare_dataset(x, idx), with_indices=True)
else:
    raise ValueError("Dataset is empty. Please check your dataset.")

# Filter out long input sequences
def is_not_too_long(example):
    input_ids = example.get('input_ids')  # Ensure 'input_ids' is correctly referenced
    return input_ids is not None and len(input_ids) < 300

dataset = dataset.filter(is_not_too_long)
print(f"Filtered dataset length: {len(dataset)}")

# Print a few examples to verify normalization
print("Normalized dataset samples:", dataset[2:4])

# Split the dataset into training and testing sets
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # Collate the inputs and targets into a batch
        batch = self.processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        del batch["decoder_attention_mask"]

        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [length - length % model.config.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

# Check the shape of a batch
if len(dataset["train"]) > 0:
    features = [
        dataset["train"][0],
        dataset["train"][1]  # Ensure this index is valid
    ]
    batch = data_collator(features)
    print({k: v.shape for k, v in batch.items()})

model.config.use_cache = False

# Use a partial function to allow caching
from functools import partial
model.generate = partial(model.generate, use_cache=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_Gujarati",  # Change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=1500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_runtime"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits), torch.tensor(labels), reduction="mean"
    )
    return {"eval_loss": loss.item()}

# Set up the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()
