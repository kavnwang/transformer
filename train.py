import itertools
import json
import os

from datasets import load_dataset, load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)

from moe_transformer import Transformer
from utils.generator import generate
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.writers import StatsWriter

model_config = json.load(open("model_config.json"))
model_config["use_cache"] = False
job_config = json.load(open("job_config.json"))

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

wandb.init(
    project="deepseek",
    entity="kavn",
    config={
        **model_config,
        **job_config,
        "device": str(device)
    }
)

print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(
    "fla-hub/gla-1.3B-100B", trust_remote_code=True
)
tokenizer.pad_token_id = job_config["pad_token_id"]

model = Transformer(**model_config).to(device)

local_dataset_path = "fineweb-edu-toy"

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
    return tokenizer(
        batch["text"], max_length=job_config["sequence_length"], truncation=True
    )


tokenized_dataset = dataset.map(
    tokenize, batched=True, remove_columns=dataset.column_names
)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(
    tokenized_dataset, batch_size=job_config["batch_size"], collate_fn=collator
)


loss_fn = nn.CrossEntropyLoss(
    ignore_index=job_config["pad_token_id"], label_smoothing=1e-5
)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.95), lr=job_config["lr"])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=job_config["warmup_steps"],
    num_training_steps=job_config["total_training_steps"]-job_config["warmup_steps"],
)

def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    if os.path.exists(job_config["model_save_path"]):
        # Raise error if the directory exists
        print("Error: Model save path already exists.")
        return
    '''
    os.makedirs(job_config["model_save_path"], exist_ok=True)
    wandb.watch(model, log="gradients", log_freq=10)
    model.train()

    with StatsWriter(
        os.path.join(job_config["model_save_path"], "stats.csv"), ["step", "loss", "lr"]
    ) as writer:
        for batch, sample in (
            bar := tqdm(
                enumerate(
                    itertools.islice(dataloader, job_config["total_training_steps"])
                ),
                total=job_config["total_training_steps"],
            )
        ):
            x = sample["input_ids"][:, :-1]
            y = sample["input_ids"][:, 1:]
            x = x.to(device)
            y = y.to(device)
            pred, _ = model(x)
            loss = loss_fn(pred.transpose(1, 2), y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            bar.set_postfix_str(f"loss: {loss.item():>7f}")
            writer.write(
                {"step": batch, "loss": loss.item(), "lr": scheduler.get_last_lr()[0]}
            )
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "step": batch
            })
            if (batch + 1) % job_config["save_interval"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step=batch,
                    checkpoint_dir=job_config["model_save_path"],
                )
                bar.write(f"Model checkpoint saved at step {batch+1}")
                sample_text = generate(model, tokenizer, model_config, "", device, False)
                bar.write(f"Sample text at step {batch+1}: {sample_text}")

train_loop(train_dataloader, model, loss_fn, optimizer)
wandb.finish()
