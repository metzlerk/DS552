import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load the training prompts from CSV
train_data = pd.read_csv("train_prompts.csv")

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(train_data)

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
  return tokenizer(examples["prompt"], text_target=examples["completion"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
  output_dir="./results",
  evaluation_strategy="epoch",
  learning_rate=5e-5,
  per_device_train_batch_size=2,
  num_train_epochs=3,
  weight_decay=0.01,
  save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")