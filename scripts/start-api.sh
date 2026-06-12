#!/bin/zsh
# Chatbot API :8000 — full PDF index, self-hosted mem0 on, telemetry off.
# Runs in the foreground; memory persists across restarts (that's the point).
BOT_DIR="${BOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHONPATH="$BOT_DIR" PDF_PATH="$BOT_DIR/pdf" CHROMA_PATH="$BOT_DIR/vectorstore" \
  OPENAI_BASE_URL="http://127.0.0.1:8080/v1" \
  MEM0_ENABLED=1 MEM0_PATH="$BOT_DIR/mem0_store" MEM0_USER_ID="${MEM0_USER_ID:-$USER}" MEM0_TELEMETRY=False \
  exec "$BOT_DIR/.venv/bin/python" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
