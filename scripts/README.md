# Launch scripts — local LLM + API with persistent memory

Scripts for running the chatbot fully locally on a large model with a compressed KV cache:

- **LLM**: Qwen3.5-35B-A3B (MoE), GGUF Q4_K_M, served by
  [TheTom's TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant)
  (branch `feature/turboquant-kv-cache`) — the fork auto-selects q8_0 K / turbo4 V for this
  model (GQA 8:1), roughly 60% smaller KV cache than f16.
- **Memory**: self-hosted [mem0](https://github.com/mem0ai/mem0) (see `memory.py`), reusing the
  same local LLM and embeddings. `MEM0_TELEMETRY=False` disables mem0's anonymous telemetry —
  nothing leaves the machine.

## Run

```bash
./scripts/start-llm.sh     # terminal 1 — wait for "server is listening"
./scripts/start-api.sh     # terminal 2 — wait for "Application startup complete"
cd frontend && npm install && npm run dev   # terminal 3 → http://localhost:5173
```

Build the fork with `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build
build -j`. Override `FORK_DIR`, `MODEL_GGUF`, `BOT_DIR`, or `MEM0_USER_ID` via environment if
your paths differ.

Toggle **Memory** in the UI (next to the RAG toggle). Facts are extracted in the background a few
seconds after each answer and persist across restarts.

## Notes that will save you an hour

- **`-c 32768` is required** when memory is on: mem0's fact-extraction prompt is ~8k+ tokens and
  silently fails at the default 8192 context.
- **`--reasoning off` is required** for reasoning models (Qwen3.x, DeepSeek-R1, …): they emit
  `reasoning_content` and leave `content` empty, which breaks both chat and memory extraction.
  The API logs a loud error at startup if you forget.
- **Resetting memory means wiping two stores**: the Chroma store (`mem0_store/`) *and*
  `~/.mem0/history.db` — mem0 keeps a cross-session message log there and will re-extract old
  facts from it. `reset-memory.sh` handles both; run it only while the API is stopped.
- Quantized V cache requires flash attention (`-fa on`).
