import torch
import torch.nn as nn
from transformer import Transformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import itertools
import json
import os
from datasets import load_from_disk

model_config = json.load(open("model_config.json"))
job_config = json.load(open("job_config.json"))

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B", trust_remote_code=True)
tokenizer.pad_token_id = job_config["pad_token_id"]

model = Transformer(**model_config).to(device)

print(model)

local_dataset_path = "fineweb-edu"

if os.path.exists(local_dataset_path) and os.path.isdir(local_dataset_path):
    dataset = load_from_disk(local_dataset_path)
else:
    dataset = load_dataset(
        job_config["dataset"],
        name=job_config["dataset_name"],
        split=job_config["split"],
        streaming=job_config["streaming"],
    )

def tokenize(batch):
    return tokenizer(batch["text"], max_length=job_config["sequence_length"], truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_dataset, batch_size=job_config["batch_size"], collate_fn=collator)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, sample in enumerate(itertools.islice(dataloader, job_config["training_steps"])):
        x = sample["input_ids"][:, :-1]
        y = sample["input_ids"][:, 1:]
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred.transpose(1, 2), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 1 == 0:
            loss, current = loss.item(), batch + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{job_config['training_steps']:>5d}]")

loss_fn = nn.CrossEntropyLoss(ignore_index=job_config["pad_token_id"])
optimizer = torch.optim.Adam(model.parameters(), lr=job_config["lr"])
train_loop(train_dataloader, model, loss_fn, optimizer)