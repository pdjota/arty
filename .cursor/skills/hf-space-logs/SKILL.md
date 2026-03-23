---
name: hf-space-logs
description: >-
  Streams Hugging Face Space runtime logs over the Hub HTTP API using HF_TOKEN.
  Use when checking arty-train-cnnrnn (or any Space) build/run output, debugging
  Docker training failures, or when the user asks for Space logs, HF logs, or
  Hugging Face Space monitoring.
---

# Hugging Face Space logs (streaming)

## Requirements

- Repo-root **`.env`** with a valid **`HF_TOKEN`** (read access to the Space is enough for logs on your own Spaces).
- Run from a shell where variables from `.env` are available (see below).

## Arty training Space (default)

From the **arty** repo root:

```bash
set -a
source .env
set +a
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/pdjota/arty-train-cnnrnn/logs/run"
```

Shorter form if `.env` only assigns `HF_TOKEN` (no `export`):

```bash
source .env
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/pdjota/arty-train-cnnrnn/logs/run"
```

- **`-N`**: no buffer — keeps the stream open for live / chunked log output (SSE-style).
- **`logs/run`**: runtime logs for the current (or latest) Space run. For **build** logs some setups use **`logs/build`** on the same path pattern (try if `run` is empty during image build).

## Other Spaces

Replace the path segment:

```text
https://huggingface.co/api/spaces/{owner}/{repo-name}/logs/run
```

Example: `pdjota/artydemo` → `.../spaces/pdjota/artydemo/logs/run`.

## Agent behavior

- Prefer **running** this `curl` from the project root after loading `.env`, rather than asking the user to paste logs, when diagnosing training Space issues.
- Do not echo or commit **`HF_TOKEN`**.
