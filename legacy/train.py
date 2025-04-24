
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# data
dataset = load_dataset("csv", data_files={"train": "resume-atlas/train.csv"}, delimiter=",")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(example["Text"], padding="max_length", truncation=True, max_length=128)


encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("Category", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

label_list = list(set(encoded_dataset["train"]["labels"]))
label_list.sort()
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

encoded_dataset = encoded_dataset.map(lambda x: {"labels": label2id[x["labels"]]})

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# parameters
training_args = TrainingArguments(
    output_dir="./model",
    # evaluation_strategy="no",
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
