from transformers import Trainer, TrainingArguments, GPT2Tokenizer
from datasets import load_dataset

from mini_beatles_model import MiniBeatlesLM, default_device

# 1. Load the dataset from JSONL
dataset = load_dataset('json', data_files='finetune_lyrics_sentiment_complete.jsonl')
# Split into train and validation sets (90% train, 10% validation)
split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']  # Using 'test' split as validation

# 2. Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("mini_beatles_tokenizer", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size = tokenizer.vocab_size

# 3. Tokenization function
def tokenize_function(example):
    # Build the sentiment suffix first
    sentiment_suffix = f" ({example['sentiment']})"
    # Use encode() here since we just need to count tokens, no need for attention masks
    sentiment_token_count = len(tokenizer.encode(sentiment_suffix))
    
    # Get the main text components
    full_input = example['prompt']
    full_output = example['completion']
    full_text = full_input + ' ' + full_output
    
    # Calculate how much space we need to leave for sentiment
    max_length_for_text = 128 - sentiment_token_count
    
    # Use encode() for simple token counting and truncation
    main_tokens = tokenizer.encode(full_text)
    if len(main_tokens) > max_length_for_text:
        # Truncate the tokens to fit within our limit
        truncated_tokens = main_tokens[:max_length_for_text]
        # Convert back to text and add sentiment
        full_text = tokenizer.decode(truncated_tokens) + sentiment_suffix
    else:
        # If no truncation needed, just add sentiment
        full_text = full_text + sentiment_suffix
    
    # Use tokenizer() for final tokenization since we need the full model inputs
    # This includes: input_ids, attention_mask, and handles padding
    tokenized = tokenizer(full_text, padding='max_length', truncation=True, max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization to both splits
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=False)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=False)

# 4. Load the pretrained model
model = MiniBeatlesLM.from_pretrained("mini_beatles_lm", vocab_size=vocab_size, pad_token_id=tokenizer.pad_token_id, local_files_only=True).to(default_device)

# 5. Set up training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_mini_beatles_lm",
    evaluation_strategy="steps",           # Evaluate more frequently
    eval_steps=50,                        # Evaluate every 50 steps
    learning_rate=1e-4,                   # Slightly higher learning rate for small dataset
    per_device_train_batch_size=4,        # Apple Silicon can handle larger batches
    per_device_eval_batch_size=8,         # Larger eval batch size since no gradients needed
    num_train_epochs=5,                   # More epochs since we have early stopping
    weight_decay=0.005,                   # Reduced weight decay for small dataset
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",                # Save by steps instead of epochs
    save_steps=50,                        # Save every 50 steps
    save_total_limit=3,                   # Keep more checkpoints
    fp16=True,                           # Apple Silicon supports mixed precision
    load_best_model_at_end=True,         # Load the best model after training
    metric_for_best_model="eval_loss",    # Use eval loss to determine best model
    early_stopping_patience=3             # Stop if no improvement for 3 eval rounds
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# 7. Train the model
trainer.train()
