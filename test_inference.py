import torch
from transformers import GPT2Tokenizer
from mini_beatles_model import MiniBeatlesLM, max_len, default_device

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
sample_text = "There's nothing you"
input_ids = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length")
input_ids = input_ids["input_ids"].to(default_device)

with torch.no_grad():
    logits = model(input_ids)
    preds = logits.argmax(-1)
    decoded = tokenizer.decode(preds[0], skip_special_tokens=True)

print(f"Input: {sample_text}")
print(f"Model output: {decoded}")
