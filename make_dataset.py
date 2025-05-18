from datasets import load_dataset
from transformers import GPT2Tokenizer
import re

# Parameters (should match your pretrain.py)
max_len = 128
window_size = max_len + 1  # input + next token
stride = 1
out_file = "beatles_lyrics.txt"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the Beatles lyrics dataset from Hugging Face
beatles_ds = load_dataset("cmotions/Beatles_lyrics", split="dataset_full")

# Tokenize once
with open(out_file, "w", encoding="utf-8") as f:
    for entry in beatles_ds:
        # Remove bracketed annotations like [Intro], [Verse 1], etc.
        lyrics = re.sub(r"\[.*?\]", "", entry["lyrics"])

        tokens = tokenizer(lyrics, truncation=False)["input_ids"]
        # Write sliding windows so that the model can learn to predict the next token
        # no matter where the input starts (e.g., "Nothing you can do that" or "There's nothing you can do that")
        for i in range(0, len(tokens) - window_size + 1, stride):
            chunk = tokens[i:i+window_size]
            text = tokenizer.decode(chunk, skip_special_tokens=True).strip()
            if text:
                f.write(text.replace("\n", " ").strip() + "\n")

print(f"Wrote sliding windowed lyrics to {out_file}")
