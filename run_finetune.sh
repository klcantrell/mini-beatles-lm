#!/bin/zsh
# Run finetuning with MPS fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 python finetune.py
