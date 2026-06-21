<p align="center">
  <img src="docs/assets/banner.png" alt="PixelRAG — Visual Retrieval-Augmented Generation" width="100%">
</p>
<p align="center">
  Official codebase for <b><a href="assets/pixelrag-paper.pdf">PIXELRAG: Web Screenshots Beat Text for
Retrieval-Augmented Generation</a></b>
</p>
<p align="center">
  <a href="https://yichuan-w.github.io/">Yichuan Wang</a>*,
  <a href="https://zhifei.li/">Zhifei Li</a>*,
  <a href="https://zwcolin.github.io/">Zirui Wang</a>,
  <a href="https://www.linkedin.com/in/paul-teiletche/">Paul Teiletche</a>,
  <a href="https://www.linkedin.com/in/lesheng-jin-9618b0201/">Lesheng Jin</a>
  <br>
  <a href="https://people.eecs.berkeley.edu/~matei/">Matei Zaharia</a>†,
  <a href="https://people.eecs.berkeley.edu/~jegonzal/">Joseph E. Gonzalez</a>†,
  <a href="https://www.sewonmin.com/">Sewon Min</a>†
</p>
<p align="center"><sub>* Equal contribution &nbsp; † Equal advising</sub><br><sub>Work done at <a href="https://sky.cs.berkeley.edu/">Berkeley SkyLab</a> &amp; <a href="https://bair.berkeley.edu/">BAIR</a> &amp; <a href="https://nlp.cs.berkeley.edu/">Berkeley NLP</a></sub></p>
<p align="center">Search any document by how it <em>looks</em>, not just the text it contains.</p>

<p align="center">
  <a href="https://github.com/StarTrail-org/PixelRAG/actions/workflows/ci.yml"><img src="https://github.com/StarTrail-org/PixelRAG/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pixelrag.ai"><img src="https://img.shields.io/badge/demo-pixelrag.ai-7c3aed" alt="Live demo"></a>
  <a href="https://status.pixelrag.ai"><img src="https://img.shields.io/badge/status-live-22c55e" alt="Status"></a>
  <a href="https://join.slack.com/t/leann-e2u9779/shared_invite/zt-3ol2ww9ic-Eg_kB8omwe6xmYVd0epr4Q"><img src="https://img.shields.io/badge/Slack-join-4A154B?logo=slack&logoColor=white" alt="Slack"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
</p>

<p align="center">
  <a href="#what-it-is">What it is</a> &middot;
  <a href="#give-claude-eyes">Give Claude eyes</a> &middot;
  <a href="#how-it-works">How it works</a> &middot;
  <a href="#pipelines">Pipelines</a>
</p>

---

```bash
pip install pixelrag
```

The two core operations — **render** a page to screenshots, **search** a visual index:

```bash
# Render any page or document to screenshot tiles
pixelshot https://en.wikipedia.org/wiki/Python --output ./tiles

# Search a hosted index of 8.28M Wikipedia pages — no setup, runs against the live API
curl -X POST https://api.pixelrag.ai/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "What is the capital of France?"}], "n_docs": 5}'
```

> **Live, hosted endpoint** — [`https://api.pixelrag.ai`](https://api.pixelrag.ai/status) serves a
> pre-built index of **8.28M Wikipedia pages**. No setup, no API key. It even takes an image as the query
> ([visual search](https://pixelrag.ai/docs#search)) — see the **[API reference →](https://pixelrag.ai/docs)**.

Or try it in the browser at **[pixelrag.ai](https://pixelrag.ai)**, or run the demo notebook in
Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StarTrail-org/PixelRAG/blob/main/demos/quickstart.ipynb) — it
renders a page and searches the hosted index, with the images inline.

## What it is

PixelRAG renders documents — web pages, PDFs, images — as screenshots and retrieves over the
images directly. Visual structure that HTML parsing throws away — tables, charts, layout,
infographics — stays intact, so the reader model can actually answer questions about it.
Wikipedia's 8.28M articles ship as a pre-built index; the pipeline itself is general-purpose.

## Give Claude eyes

The renderer also ships as a Claude Code plugin — the **pixelbrowse** skill. Instead of fetching
raw HTML, Claude screenshots a page with `pixelshot` and _reads the image_, so it sees
charts, diagrams, tables, and layout the way a person does.

Install it — no clone needed (`pixelshot` comes from `pip install pixelrag`):

```bash
pip install pixelrag                                # provides the pixelshot command
claude plugin marketplace add StarTrail-org/PixelRAG
claude plugin install pixelbrowse@pixelrag-plugins
```

Then just ask Claude to look at a page:

```bash
claude -p "screenshot https://news.ycombinator.com and summarize the top stories"
claude -p "screenshot https://arxiv.org/abs/2404.12387 and explain the key findings"
```

Or use the slash command in an interactive session: `/screenshot https://example.com`.
No MCP server, no backend: the skill just calls `pixelshot` (Playwright/CDP) on your machine.

## How it works

<p align="center">
  <img src="docs/assets/pipeline.png" alt="Text-based RAG parses to text and loses the table; PixelRAG renders to screenshot tiles and keeps it" width="100%">
</p>

Text-based RAG parses the page to text chunks and **loses the table** — the reader can't find the
answer. PixelRAG renders the page to **screenshot tiles**, retrieves the right tile, and the reader
reads the number straight off the image.

Two pieces make this work: (1) rendering documents to images instead of parsing them to text, and
(2) a `Qwen3-VL-Embedding` model, LoRA-fine-tuned on screenshot data, that embeds page images into
a space where visual content is retrievable.

## Pipelines

Capture is the standalone `pixelshot` command; the rest of the pipeline runs through the
`pixelrag` umbrella — `pixelrag <stage>`. Install only the stages you need:

| Command                                    | What it does                                                    | Install                         |
| ------------------------------------------ | --------------------------------------------------------------- | ------------------------------- |
| `pixelshot`                                | Document → image tiles (Playwright CDP, PDF)                    | `pip install pixelrag`          |
| `pixelrag chunk` · `embed` · `build-index` | Tiles → vectors → FAISS index                                   | `pip install 'pixelrag[embed]'` |
| `pixelrag index`                           | Orchestrates the full pipeline: source → ingest → embed → index | `pip install 'pixelrag[index]'` |
| `pixelrag serve`                           | FAISS search API (FastAPI, CPU or GPU)                          | `pip install 'pixelrag[serve]'` |

```
render ←── index ──→ embed       serve (independent)       train → serve (HTTP)
```

**`train` is a separate uv project** with its own pinned env (`torch==2.9.1+cu129`,
`transformers==4.57.1`, cuDNN 9.20) — install it from inside `train/`, not from the root.

### Search a pre-built index

```bash
pip install 'pixelrag[serve]'

# Download a pre-built index from Hugging Face. The dataset repo holds four FAISS indexes
# (base/LoRA Wikipedia pixel, Wikipedia text, news pixel); grab just the base one (~217G) here.
huggingface-cli download StarTrail-org/pixelrag-faiss-indexes \
  --repo-type dataset --include "search_index_normed_v2/*" --local-dir ./index

# Serve, then query
pixelrag serve --index-dir ./index/search_index_normed_v2 --port 30001

curl -X POST http://localhost:30001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "What is the capital of France?"}], "n_docs": 5}'
```

### Build an index from your own documents

Works on **Linux (CUDA)** and **macOS (Apple Silicon / MPS)** — `device: auto` picks the best backend.

```bash
pip install 'pixelrag[index]'

# Create pixelrag.yaml
cat > pixelrag.yaml << 'EOF'
source:
  type: local
  path: ./my_docs

embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: auto          # cuda on Linux, mps on macOS, cpu as fallback

output: ./my_index
EOF

# Build, then serve
pixelrag index build
pixelrag serve --index-dir ./my_index --port 30001
```

<details>
<summary><strong>Try it: index a PDF and search it locally</strong></summary>

No GPU required — runs on macOS (Apple Silicon) or any machine with Python 3.10+.

```bash
pip install 'pixelrag[index]'

# 1. Grab a sample PDF (or use your own)
curl -L -o paper.pdf https://raw.githubusercontent.com/StarTrail-org/PixelRAG/main/assets/pixelrag-paper.pdf

# 2. Create config (device: auto picks MPS on Mac, CUDA on Linux)
cat > pixelrag.yaml << 'EOF'
source:
  type: local
  path: ./paper.pdf

embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: auto

output: ./paper_index
EOF

# 3. Build the index (~3 min on Apple M-series, ~1 min on GPU)
pixelrag index build

# 4. Serve it
pixelrag serve --index-dir ./paper_index --port 30001

# 5. Search — should return page 2 (the overview diagram)
curl -X POST http://localhost:30001/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "Overview of PixelRAG and the diagram"}], "n_docs": 1}'
```

</details>

### Render a page programmatically

```python
from pixelrag_render import render_url

# render a single page to tiles — e.g. for an agent to read
tiles = render_url("https://en.wikipedia.org/wiki/Python", "./tiles")
```

### Embed tools (standalone)

Each stage runs independently, without the orchestrator:

```bash
pip install 'pixelrag[embed]'

pixelrag chunk --tiles-dir ./tiles
pixelrag embed --shard-dir ./tiles --output-dir ./embeddings --gpu-ids 0,1
pixelrag build-index --embeddings-dir ./embeddings --output-dir ./index
```

### Training

Fine-tuning lives in `train/` — a **separate uv project** (`wiki-screenshot-training`) with its own
pinned env. It LoRA-fine-tunes `Qwen/Qwen3-VL-Embedding-2B` for webpage retrieval; run it from
inside `train/` (`cd train && uv sync`). See [`train/README.md`](train/README.md) for the full recipe.

You don't need to retrain to use the model — the trained adapters are published at
[`Chrisyichuan/wiki-screenshot-embedding-lora`](https://huggingface.co/Chrisyichuan/wiki-screenshot-embedding-lora/tree/main/lora_vit/ckpt200).

We also release the full training set
([`Chrisyichuan/screenshot-training-natural-filtered-v2`](https://huggingface.co/datasets/Chrisyichuan/screenshot-training-natural-filtered-v2)),
so you can adapt other backbones yourself — a larger Qwen, or any other embedding model.
The data curation pipeline (LLM-augmented query generation, filtering, hard-negative mining)
is documented in [`train/docs/synthetic_data_pipeline.md`](train/docs/synthetic_data_pipeline.md).

## Acknowledgments

Thanks to [Rulin Shao](https://rulinshao.github.io/) for support.

Thanks also to [Claude Code](https://github.com/anthropics/claude-code) and
[OpenAI Codex](https://github.com/openai/codex) for supporting open-source contributors with credits and plans,
which we earned by working on [LEANN](https://github.com/StarTrail-org/LEANN).

This work is done by the [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/),
[BAIR](https://bair.berkeley.edu/), and the [Berkeley NLP Group](https://nlp.cs.berkeley.edu/).

## License

Apache-2.0
