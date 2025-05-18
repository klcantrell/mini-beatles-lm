import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.optim.lr_scheduler import LambdaLR  # Added import for learning rate scheduler
from mini_beatles_model import MiniBeatlesLM, max_len, default_device, embed_dim, n_heads, n_layers, ff_dim

# 1. Hyperparameters
batch_size  = 8
lr          = 5e-4
n_epochs    = 50
device      = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 2. Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size = tokenizer.vocab_size

# 3. Dataset & DataLoader
ds = load_dataset("text", data_files={"train": "beatles_lyrics.txt"})["train"]
def encode(ex):
    toks = tokenizer(ex["text"], truncation=True, max_length=max_len+1, padding="max_length")
    return {"input_ids": torch.tensor(toks["input_ids"], dtype=torch.long)}
ds = ds.map(encode, remove_columns=["text"])
ds.set_format(type="torch", columns=["input_ids"])
loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 4. Model
model = MiniBeatlesLM(vocab_size, tokenizer.pad_token_id).to(default_device)

# Test untrained model output
# sample_text = "There's nothing you"
# input_ids = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length")["input_ids"].to(device)
# with torch.no_grad():
#     logits = model(input_ids)
#     preds = logits.argmax(-1)
#     decoded = tokenizer.decode(preds[0], skip_special_tokens=True)
# print(f"Input: {sample_text}")
# print("Untrained model output:", decoded)

# 7. Training setup
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Add a learning rate scheduler with warmup
warmup_steps = 100
num_training_steps = n_epochs * len(loader)
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
    )
scheduler = LambdaLR(optimizer, lr_lambda)

model.train()
for epoch in range(1, n_epochs + 1):
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(default_device)  # (B, T+1)
        # Shifted labels: inputs are input_ids[:, :-1], labels are input_ids[:, 1:]
        # This is necessary for next-token prediction: the model sees the first N tokens as input,
        # and is trained to predict the next token at each position. This is standard for language modeling.
        # For example, given input "There's nothing you", the model should learn to predict the next word
        # ("can") at the next position, and so on for each subsequent token in the sequence.
        inputs = input_ids[:, :-1]  # (B, T)
        labels = input_ids[:, 1:]   # (B, T)
        logits = model(inputs)      # (B, T, V)
        loss = criterion(logits.view(-1, vocab_size), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        # This clips the gradients to a maximum norm of 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Step the learning rate scheduler
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} ▶︎ Avg loss: {total_loss/len(loader):.4f}")

# 8. Save
torch.save(model.state_dict(), "mini_beatles_llm.pth")
tokenizer.save_pretrained("mini_beatles_tokenizer")
