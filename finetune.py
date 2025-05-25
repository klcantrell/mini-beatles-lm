from transformers import Trainer, TrainingArguments, GPT2Tokenizer, EarlyStoppingCallback
from datasets import load_dataset

from mini_beatles_model import MiniBeatlesLM, default_device


# 1. Load the dataset from JSONL
dataset = load_dataset('json', data_files='finetune_lyrics_with_emojis.jsonl')
# Split into train and validation sets (90% train, 10% validation)
split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']  # Using 'test' split as validation

# 2. Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("mini_beatles_tokenizer", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# Add heart emoji to tokenizer
tokenizer.add_tokens(['❤️'])  # Always add the token to ensure consistent token ID
vocab_size = len(tokenizer)

# 3. Tokenization function
def tokenize_function(example):
    # First, get the input encoding
    input_encoding = tokenizer(
        example['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Create labels starting with all -100 (ignored in loss calculation)
    input_ids = input_encoding["input_ids"][0].tolist()
    labels = [-100] * len(input_ids)
    
    # Get the heart emoji token ID
    heart_token_id = tokenizer.convert_tokens_to_ids('❤️')
    
    # For each position where the heart emoji appears, set labels to predict it
    for i, token_id in enumerate(input_ids):
        if token_id == heart_token_id:
            labels[i] = token_id
    
    # Convert to regular Python types for dataset storage
    return {
        "input_ids": input_ids,
        "attention_mask": input_encoding["attention_mask"][0].tolist(),
        "labels": labels,
    }

# Apply tokenization to both splits
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=False)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=False)

# 4. Load the pretrained model with original vocab size and resize for new tokens
model = MiniBeatlesLM.from_pretrained(
    "mini_beatles_lm",
    vocab_size=50257,  # Start with original vocab size
    pad_token_id=tokenizer.pad_token_id,
    local_files_only=True
)
print(f"Original vocab size: {model.tok_emb.num_embeddings}")
print(f"New vocab size: {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))
model = model.to(default_device)

# 5. Set up training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_mini_beatles_lm",
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]  # Enabled early stopping
)

# 7. Train the model
trainer.train()
