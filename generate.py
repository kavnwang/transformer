import json
import torch
from transformers import AutoTokenizer

from moe_transformer import Transformer


temperature = 1.0

model_config = json.load(open("model_config.json"))

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
)

tokenizer = AutoTokenizer.from_pretrained(
    "fla-hub/gla-1.3B-100B", trust_remote_code=True
)

model = Transformer(**model_config).to(device)
model.load_state_dict(
    torch.load("models/test/model_checkpoint_step_4300.pth", map_location=device, weights_only=True)
)

prompt = input("Enter your prompt: ")
inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
with torch.no_grad():
    for _ in range(100):
        outputs = model(inputs)
        next_token_logits = outputs[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        inputs = torch.cat([inputs, next_token], dim=-1)
        print(tokenizer.decode(next_token.squeeze().cpu()), end="", flush=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
