import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mini_beatles_model import MiniBeatlesLM, max_len, default_device

# 1. Hyperparameters
batch_size  = 8
lr          = 5e-4
n_epochs    = 20
device      = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 2. Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size = tokenizer.vocab_size

# 3. Dataset & DataLoader
# ds = load_dataset("text", data_files={"train": "beatles_lyrics.txt"})["train"]
# def encode(ex):
#     toks = tokenizer(ex["text"], truncation=True, max_length=max_len, padding="max_length")
#     return {"input_ids": torch.tensor(toks["input_ids"], dtype=torch.long)}
# ds = ds.map(encode, remove_columns=["text"])
# ds.set_format(type="torch", columns=["input_ids"])
# loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 4. Model
model = MiniBeatlesLM(vocab_size, tokenizer.pad_token_id).to(default_device)

# Test untrained model output
sample_text = "There's nothing you"
input_ids = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length")["input_ids"].to(device)
with torch.no_grad():
    logits = model(input_ids)
    preds = logits.argmax(-1)
    decoded = tokenizer.decode(preds[0], skip_special_tokens=True)
print(f"Input: {sample_text}")
print("Untrained model output:", decoded)

# 7. Training setup
# optimizer = optim.AdamW(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# model.train()
# for epoch in range(1, n_epochs + 1):
#     total_loss = 0.0
#     for batch in loader:
#         input_ids = batch["input_ids"].to(default_device)  # (B, T)
#         labels = input_ids.clone()
#         logits = model(input_ids)                  # (B, T, V)
#         loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch} ▶︎ Avg loss: {total_loss/len(loader):.4f}")

# 8. Save
# torch.save(model.state_dict(), "mini_beatles_llm.pth")
# tokenizer.save_pretrained("mini_beatles_tokenizer")
