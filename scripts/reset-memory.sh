#!/bin/zsh
# Wipe the mem0 memory completely. Run ONLY while the API (start-api.sh) is stopped.
# Wipes BOTH stores — the Chroma store AND ~/.mem0/history.db: mem0 keeps a
# cross-session message log there and re-extracts old facts from it if you forget.
BOT_DIR="${BOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if pgrep -f 'uvicorn api.main:app' >/dev/null; then
  echo "ERROR: API still running — stop it first." >&2
  exit 1
fi
rm -rf "$BOT_DIR/mem0_store" ~/.mem0/history.db
echo "Memory wiped. Restart the API and reload the browser page."
