import torch
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained(
    "fla-hub/gla-1.3B-100B", trust_remote_code=True
)

def generate(model, model_config, prompt, device, use_cache, max_length: int = 100, temperature: float = 1.0) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    if use_cache:
        key_cache = [None for _ in range(model_config["num_layers"])]
        value_cache = [None for _ in range(model_config["num_layers"])]
    with torch.no_grad():
        for _ in range(max_length):
            if use_cache:
                outputs, (key_cache, value_cache) = model(inputs,key_cache,value_cache)
            else: 
                outputs, _ = model(inputs)
            next_token_logits = outputs[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            inputs = torch.cat([inputs, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(inputs[0])
