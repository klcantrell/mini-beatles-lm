#!/bin/zsh
# Run the Beatles lyrics generator with MPS fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_inference.py "$@"
