#!/usr/bin/env python3
from datasets import load_dataset
import os
import shutil

output_dir = "fineweb-edu"

os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split=f"train",
    trust_remote_code=True
)

dataset.save_to_disk(output_dir)

