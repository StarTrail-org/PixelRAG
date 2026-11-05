# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# PixelRAG

Visual RAG: render documents (web pages, PDFs) as screenshot tiles and retrieve over the images
with a Qwen3-VL embedding model. See `README.md` for the product overview and `deploy/README.md`
for how it runs in production.

## Layout
- `render/` — `pixelshot` capture (Playwright/CDP, PDF)
- `embed/` — tiles → chunks → vectors → index (the `chunk`, `embed`, `build-index` stages)
- `index/` — `pixelrag index` orchestrator + pluggable document sources
- `serve/` — FAISS/Qdrant search API (`pixelrag serve`)
- `web/` — Next.js frontend (on Vercel) + `agent-server.mjs` (the chat agent backend)
- `train/` — **separate uv project** (LoRA finetune); install from inside `train/`, not the root
- `eval/` — **separate uv project** reproducing paper Table 1 (reader + grader over HTTP)
- `deploy/` — systemd units, CD workflow, blue-green scripts

## Commands

Python (uv only — `uv add`, never `uv pip install`; work in `.venv`; commit `uv.lock`):

```bash
# Install dev deps and run the full Python suite (render tests need Chrome too)
uv sync --extra dev
uv run pixelshot install-chrome        # one-time; CI also runs `npx -y playwright install-deps chromium`
uv run pytest tests/ -v
uv run pytest tests/test_cli.py::test_<name> -v   # single test

# Lint / format (ruff; config in pyproject.toml excludes train/, demos/, docs/, skill/)
uvx ruff check .
uvx ruff format --check .

# Pipeline: build an index from a pixelrag.yaml config, then serve it
pixelrag index build -c pixelrag.yaml
pixelrag serve --index-dir ./index --port 30001
```

Frontend (`cd web`):

```bash
npm ci
npm test            # node --test lib/*.test.mjs
npm run build
npm run typecheck   # tsc --noEmit
```

## Architecture

**One distribution, many packages.** `pixelrag` is the single PyPI package (`pyproject.toml`
bundles `src/pixelrag` plus `render/embed/index/serve` under `src/`). Two entry points: the
standalone `pixelshot` (`pixelrag_render.render:main`) and the `pixelrag` umbrella
(`pixelrag.cli:main`). Heavy ML stages are opt-in extras (`embed`, `index`, `serve`, `qdrant`, `pdf`).

**CLI dispatch is lazy.** `src/pixelrag/cli.py` maps `pixelrag <stage>` to a stage module and
imports it on demand, printing an install hint if the extra is missing. Stage 0 (render) is the
separate `pixelshot` command, kept torch-free in core.

**`pixelrag index` orchestrates by subprocess.** `index/src/pixelrag_index/pipelines.py:build`
walks `source → ingest (render tiles) → chunk → embed → build-index`, shelling out to
`python -m pixelrag_embed.chunk / .embed_cpu / .embed / .index` rather than importing them.
Device routing happens here: `device: cpu|mps|auto` → `embed_cpu.py`, `cuda` → `embed.py`
(vLLM/sglang).

**Tile data model.** Each document gets a numeric position index. Tiles live in
`<output>/tiles/<idx>.png.tiles/` containing `tile_XXXX.jpg`, a `tiles.json` manifest
(`article_id`, `source`, `page_height`, `tiles`, `complete`), and — after chunking —
`chunks.json` + `chunk_XXXX_YY.png` (≤1024px strips). `embed` reads `article_id` from the
manifest, never from the directory name. `<output>/articles.json` maps position index →
title/url/department (department = first subdirectory under a local source root, used for
filtered search). Manifests are written atomically (`*.tmp` + `os.replace`) and `_needs_render`
uses the `source` field to detect stale tile dirs when the source set changes between runs.

**Config.** `pixelrag.yaml` is deep-merged over `DEFAULT_CONFIG` in
`index/src/pixelrag_index/config.py` (`ingest`, `embed` sections merge key-wise). Sources are
pluggable via the `SOURCES` registry in `index/src/pixelrag_index/sources/__init__.py`
(`local`, `web`, `kiwix`, `pdf`); each yields `Document(id, url, path, metadata)`.

**Serve.** `serve/src/pixelrag_serve/api.py` is a FastAPI app serving FAISS (default) or Qdrant
backends, with text and image queries. It also holds `render_ondemand.py` (render retrieved
pages at query time from a kiwix ZIM) and `zim_server*.py`. `web/agent-server.mjs` calls
`/search`; its `/health` is liveness and `/ready` is readiness (see `deploy/README.md`).

**Training and eval are separate uv projects.** `train/` LoRA-fine-tunes
`Qwen/Qwen3-VL-Embedding-2B` (`train_contrastors.py` is the primary script); it has its own
`train/CLAUDE.md` and `train/README.md` — read those before touching it. `eval/` is a pure HTTP
client (reader + grader) that reproduces paper numbers; also self-contained.

## Conventions
- `main` is branch-protected — land changes via PRs.
- Releasing is automated: `uv version X.Y.Z`, tag `vX.Y.Z`, GitHub Release → `.github/workflows/release.yml` publishes `pixelrag` to PyPI.

## Operational context (host-specific, kept out of this public repo)
Deploy/runtime details that are specific to the live host are **not** committed here. They are
imported below; the import is a harmless no-op anywhere the file doesn't exist:

@~/.claude/pixelrag-ops.md

If deploy-host notes appeared from that import, **you are on the deploy host** — production
services run there, so operate carefully. If nothing appeared, you're on a normal dev checkout
and can ignore deployment concerns.
