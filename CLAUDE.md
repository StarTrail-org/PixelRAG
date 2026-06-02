# PixelRAG

Visual RAG: render documents (web pages, PDFs) as screenshot tiles and retrieve over the images
with a Qwen3-VL embedding model. See `README.md` for the product overview and `deploy/README.md`
for how it runs in production.

## Layout
- `render/` — `pixelshot` capture (Playwright/CDP, PDF)
- `embed/`, `index/` — tiles → vectors → FAISS index
- `serve/` — FAISS search API (`pixelrag serve`)
- `web/` — Next.js frontend (on Vercel) + `agent-server.mjs` (the chat agent backend)
- `train/` — **separate uv project** (LoRA finetune); install from inside `train/`, not the root
- `deploy/` — systemd units, CD workflow, blue-green scripts

## Conventions
- Python: `uv` only (`uv add`, never `uv pip install`); work in `.venv`; commit `uv.lock`.
- One distribution (`pixelrag`) with extras; the CLIs are `pixelshot` and `pixelrag <stage>`.
- `main` is branch-protected — land changes via PRs.

## Operational context (host-specific, kept out of this public repo)
Deploy/runtime details that are specific to the live host are **not** committed here. They are
imported below; the import is a harmless no-op anywhere the file doesn't exist:

@~/.claude/pixelrag-ops.md

If deploy-host notes appeared from that import, **you are on the deploy host** — production
services run there, so operate carefully. If nothing appeared, you're on a normal dev checkout
and can ignore deployment concerns.
