import torch
import torch.nn as nn
from transformer import Transformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import json


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B", trust_remote_code=True)

config = json.load(open("config.json"))
model = Transformer(**config).to(device)

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

train_dataloader = DataLoader(dataset, batch_size=64)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, sample in enumerate(dataloader):
        x = sample["input_ids"]
        y = sample["labels"]
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred.transpose(1, 2), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(sample["input_ids"])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 20
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
