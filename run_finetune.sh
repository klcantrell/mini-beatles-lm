#!/bin/zsh
# Run finetuning with MPS fallback
rm -rf finetuned_mini_beatles_lm
PYTORCH_ENABLE_MPS_FALLBACK=1 python finetune.py
