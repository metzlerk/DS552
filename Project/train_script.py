import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset
from sklearn.model_selection import train_test_split
from evaluate import load
import torch
import os
import numpy as np

# Load the training prompts from CSV
train_data = pd.read_csv("train_prompts.csv")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.1)

# Convert the DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the input and target text
    model_inputs = tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",  # Ensure consistent sequence length
        max_length=128,        # Adjust max_length based on your dataset
    )
    # Tokenize the target text (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["completion"],
            truncation=True,
            padding="max_length",  # Ensure labels match input length
            max_length=128,        # Same max_length as inputs
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save checkpoints
    evaluation_strategy="epoch",  # Evaluate and save at the end of every epoch
    save_strategy="epoch",  # Save the model at the end of every epoch
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=5,  # Increased for small datasets
    weight_decay=0.01,
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_dir="./logs",  # Directory for logs
    logging_steps=50,  # Log every 50 steps
    fp16=True,  # Enable mixed precision for faster training (if using GPU)
)

# Load BLEU for evaluation
bleu = load("sacrebleu")

# Define a function to compute metrics
def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Debug: Print the type and structure of predictions and labels
    print(f"Type of predictions: {type(predictions)}")
    print(f"Type of labels: {type(labels)}")
    print(f"First prediction: {predictions[0] if isinstance(predictions, (list, np.ndarray)) else 'Not a list or ndarray'}")
    print(f"First label: {labels[0] if isinstance(labels, (list, np.ndarray)) else 'Not a list or ndarray'}")

    # Convert logits to token IDs
    if isinstance(predictions, np.ndarray):
        predictions = np.argmax(predictions, axis=-1)  # Take the token with the highest probability
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()  # Convert labels to lists

    # Convert predictions to lists
    predictions = predictions.tolist()

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Wrap decoded_labels in a list for BLEU
    decoded_labels = [[label] for label in decoded_labels]

    return bleu.compute(predictions=decoded_preds, references=decoded_labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the final fine-tuned model
final_model_dir = "./fine_tuned_model"
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# Load the fine-tuned model for testing
generator = pipeline("text-generation", model=final_model_dir, tokenizer=final_model_dir)

# Test the model
prompt = "Generate a D&D encounter for a party of level 5 in a forest terrain with a mystery theme."
response = generator(prompt, max_length=200, num_return_sequences=1)
print(response[0]["generated_text"])