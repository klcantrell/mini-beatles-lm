import sys
import os
import warnings

from transformers import GPT2Tokenizer
from mini_beatles_model import MiniBeatlesLM, default_device, generate

# Suppress PyTorch MPS warnings until PyTorch fully supports MPS
warnings.filterwarnings("ignore", message=".*The operator.*MPS backend.*")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("mini_beatles_tokenizer", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size = tokenizer.vocab_size

# 4. Load the pretrained model
# Find the latest checkpoint in the finetuned_mini_beatles_lm directory
checkpoint_dir = "finetuned_mini_beatles_lm"
checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading model from latest checkpoint: {model_path}")
else:
    raise RuntimeError("No checkpoint found in 'finetuned_mini_beatles_lm'. Please ensure a checkpoint exists before running finetune.py.")

model = MiniBeatlesLM.from_pretrained(model_path, vocab_size=vocab_size, pad_token_id=tokenizer.pad_token_id, local_files_only=True).to(default_device)
model.eval()

# Print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Get prompt from command line argument or use default
default_prompt = "There's nothing you"
prompt = sys.argv[1] if len(sys.argv) > 1 else default_prompt
model.eval()
gen_text = generate(model, tokenizer, prompt, max_tokens=50)

print(f"\nInput: {prompt}")
print(f"Model output: {gen_text}\n")
