#!/usr/bin/env python3
from datasets import load_dataset
import os
import shutil

output_dir = "fineweb-edu-toy"
num_samples = 50000

os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    trust_remote_code=True
)

# Limit to first 10000 samples
dataset = dataset.select(range(min(num_samples, len(dataset))))

dataset.save_to_disk(output_dir)

