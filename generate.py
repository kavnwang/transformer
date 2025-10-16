import argparse
import json

import torch

from moe_transformer import Transformer
from utils.checkpointing import load_checkpoint
from utils.generator import generate


def main(model_config, device, checkpoint, max_length, temperature=1.0):
    model = Transformer(**model_config).to(device)
    load_checkpoint(model, None, None, checkpoint, device)

    prompt = input("Enter your prompt: ")
    output = generate(model, prompt, device, max_length, temperature)
    print("Generated text:")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer model."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.json",
        help="Path to the model configuration file (JSON format).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/test/checkpoint_step_14500.pth",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for text generation.",
    )
    args = parser.parse_args()

    model_config = json.load(open(args.model_config))
    device = torch.device(args.device)

    main(model_config, device, args.checkpoint, args.max_length, args.temperature)
