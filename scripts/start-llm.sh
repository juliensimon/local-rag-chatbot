#!/bin/zsh
# llama-server :8080 — Qwen3.5-35B-A3B MoE on the TurboQuant llama.cpp fork.
# The fork auto-selects q8_0 K / turbo4 V for this model (GQA 8:1) — see the
# "auto-asymmetric" line in the startup log.
FORK_DIR="${FORK_DIR:-$HOME/Development/repos/llama-cpp-turboquant}"
MODEL_GGUF="${MODEL_GGUF:-$FORK_DIR/models/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf}"
exec "$FORK_DIR/build/bin/llama-server" \
  -m "$MODEL_GGUF" \
  -fa on -ctk turbo4 -ctv turbo4 -ngl 99 -c 32768 \
  --host 127.0.0.1 --port 8080 --jinja --reasoning off
