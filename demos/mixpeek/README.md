# Managed Visual Web Search with Mixpeek

Run PixelRAG-style visual search **without local GPU, FAISS, or Playwright** — [Mixpeek](https://mixpeek.com) handles rendering, embedding, indexing, and serving.

## How it maps

| PixelRAG stage | Local pipeline | Mixpeek equivalent |
|---|---|---|
| `pixelshot` (render) | Playwright CDP → screenshot tiles | `web_scraper` extractor (`render_strategy: auto`) |
| `pixelrag chunk` | Split tiles into 1024px strips | Semantic chunking (`chunk_strategy: paragraphs`) |
| `pixelrag embed` | Qwen3-VL 2B on local GPU | SigLIP (visual) + DINOv2 (structure) + E5 (text) |
| `pixelrag build-index` | FAISS IVFFlat | Managed vector store |
| `pixelrag serve` | FastAPI search API | Retriever endpoint with RRF fusion |

The SigLIP + DINOv2 embeddings capture visual appearance the way Qwen3-VL screenshot embeddings do, but on managed infrastructure — no `pip install 'pixelrag[embed]'`, no CUDA.

## Quick start

```bash
pip install requests
export MIXPEEK_API_KEY=sk_...   # free key at https://studio.mixpeek.com

python web_search.py \
    --url https://en.wikipedia.org/wiki/Python \
    --query "type system"
```

Deeper crawl:

```bash
python web_search.py \
    --url https://docs.example.com \
    --query "rate limits" \
    --max-depth 3 --max-pages 100
```

LLM-guided crawl (follows relevant links first):

```bash
python web_search.py \
    --url https://docs.example.com \
    --query "OAuth flow" \
    --crawl-mode semantic --crawl-goal "Find auth documentation"
```

## What it does

1. Creates a throwaway Mixpeek namespace (auto-expires in 6h — no cleanup)
2. Uploads your seed URL
3. Creates a collection with the `web_scraper` extractor — crawls the site and generates text, code, image, and visual structure embeddings per page
4. Waits for processing
5. Builds a multimodal retriever (text + visual structure with RRF fusion)
6. Runs your search query and prints results

## Requirements

- Python 3.8+
- `requests` (`pip install requests`)
- A Mixpeek API key (free tier available at [studio.mixpeek.com](https://studio.mixpeek.com))

## When to use this vs. local PixelRAG

| | Local PixelRAG | Mixpeek managed |
|---|---|---|
| **Latency** | ~5 min to embed 100 articles on CPU | ~2 min (managed GPU) |
| **Hardware** | Needs GPU for fast embedding | API calls only |
| **Index size** | Local FAISS, you manage storage | Managed vector store |
| **Custom model** | Full control (LoRA fine-tune) | Fixed extractors |
| **Offline** | Works offline after setup | Requires internet |

Use local PixelRAG when you need a custom LoRA, offline operation, or full pipeline control. Use Mixpeek when you want a managed service and don't want to run inference infrastructure.
