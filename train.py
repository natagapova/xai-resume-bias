
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# data
dataset = load_dataset("csv", data_files={"train": "resume-atlas/train.csv"}, delimiter=",")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=encoded_dataset["train"].features["labels"].num_classes)

# parameters
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
)

# training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./model")
