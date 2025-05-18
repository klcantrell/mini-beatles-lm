from datasets import load_dataset
from transformers import GPT2Tokenizer
import re

# Parameters (should match your pretrain.py)
max_len = 128
out_file = "beatles_lyrics.txt"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the Beatles lyrics dataset from Hugging Face
beatles_ds = load_dataset("cmotions/Beatles_lyrics", split="dataset_full")

with open(out_file, "w", encoding="utf-8") as f:
    for entry in beatles_ds:
        lyrics = entry["lyrics"]
        # Remove bracketed annotations like [Intro], [Verse 1], etc.
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        # Tokenize and split lyrics into chunks of max_len
        tokens = tokenizer(lyrics, truncation=False)["input_ids"]
        for i in range(0, len(tokens), max_len):
            chunk = tokens[i:i+max_len]
            text = tokenizer.decode(chunk, skip_special_tokens=True).strip()
            if text:
                f.write(text.replace("\n", " ").strip() + "\n")

print(f"Wrote split lyrics to {out_file}")
