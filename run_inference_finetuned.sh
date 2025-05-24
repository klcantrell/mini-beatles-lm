#!/bin/zsh
# Run the finetuned Beatles lyrics generator with MPS fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_inference_finetuned.py "$@"
