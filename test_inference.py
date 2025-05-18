# PYTORCH_ENABLE_MPS_FALLBACK=1 python test_inference.py to run this script on macOS
# until PyTorch fully supports MPS

import torch
from transformers import GPT2Tokenizer
from mini_beatles_model import MiniBeatlesLM, default_device, generate

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("mini_beatles_tokenizer")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size = tokenizer.vocab_size

# Load model
model = MiniBeatlesLM(vocab_size, tokenizer.pad_token_id).to(default_device)
model.load_state_dict(torch.load("mini_beatles_llm.pth", map_location=default_device))
model.eval()

# Print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Inference on "There's nothing you"
prompt = "There's nothing you"
model.eval()
gen_text = generate(model, tokenizer, prompt, max_tokens=50)

print(f"Input: {prompt}")
print(f"Model output: {gen_text}")
