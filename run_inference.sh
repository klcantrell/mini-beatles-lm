#!/bin/zsh
# Run the Beatles lyrics generator with MPS fallback and warnings suppressed
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_inference.py "$@"
