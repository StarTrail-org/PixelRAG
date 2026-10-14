#!/usr/bin/env python3
"""PixelRAG-style visual web search — managed backend via Mixpeek.

Replaces the local pipeline (pixelshot → chunk → embed → FAISS → serve)
with Mixpeek's managed web_scraper extractor, which produces:
  - Text embeddings   (E5-Large, 1024D)
  - Code embeddings   (Jina Code, 768D)
  - Image embeddings  (SigLIP, 768D)    ← visual content, like Qwen3-VL
  - Structure embeddings (DINOv2, 768D)  ← visual layout similarity

No local GPU, no FAISS, no Playwright — the extraction and indexing run
on Mixpeek's infrastructure. You get a retriever endpoint you can query
with text or images.

Usage:
    export MIXPEEK_API_KEY=sk_...       # https://studio.mixpeek.com
    pip install requests

    python web_search.py --url https://en.wikipedia.org/wiki/Python --query "type system"
    python web_search.py --url https://docs.example.com --query "auth" --max-depth 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import requests

DEFAULT_BASE_URL = "https://api.mixpeek.com"
FIELD = "content"
TTL_HOURS = 6


def _session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )
    return s


def _post(s: requests.Session, url: str, body: dict, **kw) -> dict:
    r = s.post(url, json=body, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code} {r.url}: {detail}")
    return r.json()


def _get(s: requests.Session, url: str) -> dict:
    r = s.get(url)
    r.raise_for_status()
    return r.json()


# ── Pipeline steps ────────────────────────────────────────────────


def create_namespace(s, base_url):
    slug = f"pixelrag-{uuid.uuid4().hex[:8]}"
    return _post(
        s,
        f"{base_url}/v1/namespaces",
        {
            "namespace_name": slug,
            "feature_extractors": [
                {"feature_extractor_name": "web_scraper", "version": "v1"}
            ],
            "ttl_hours": TTL_HOURS,
        },
    )


def create_bucket(s, base_url, namespace_id):
    s.headers["X-Namespace"] = namespace_id
    return _post(
        s,
        f"{base_url}/v1/buckets",
        {
            "bucket_name": f"web-{uuid.uuid4().hex[:6]}",
            "bucket_schema": {"properties": {FIELD: {"type": "text"}}},
        },
    )


def upload_url(s, base_url, bucket_id, url):
    return _post(
        s,
        f"{base_url}/v1/buckets/{bucket_id}/objects",
        {"blobs": [{"property": FIELD, "type": "text", "data": url}]},
    )


def create_collection(
    s,
    base_url,
    bucket_id,
    *,
    max_depth=2,
    max_pages=50,
    crawl_mode="deterministic",
    crawl_goal=None,
    render_strategy="auto",
    chunk_strategy="paragraphs",
):
    params = {
        "extractor_type": "web_scraper",
        "max_depth": max_depth,
        "max_pages": max_pages,
        "crawl_mode": crawl_mode,
        "render_strategy": render_strategy,
        "chunk_strategy": chunk_strategy,
    }
    if crawl_goal:
        params["crawl_goal"] = crawl_goal
    return _post(
        s,
        f"{base_url}/v1/collections",
        {
            "collection_name": f"pixelrag-col-{uuid.uuid4().hex[:6]}",
            "feature_extractor": {
                "feature_extractor_name": "web_scraper",
                "version": "v1",
                "parameters": params,
                "input_mappings": {"url": FIELD},
            },
            "source": {"type": "bucket", "bucket_ids": [bucket_id]},
        },
    )


def submit_batch(s, base_url, bucket_id, collection_id, object_id):
    return _post(
        s,
        f"{base_url}/v1/buckets/{bucket_id}/batches?auto_submit=true",
        {"collection_id": collection_id, "object_ids": [object_id]},
    )


def wait_for_batch(s, base_url, bucket_id, batch_id, timeout=600, poll=10):
    print(f"\n  Waiting for batch {batch_id} …", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _get(s, f"{base_url}/v1/buckets/{bucket_id}/batches/{batch_id}")
        status = r.get("status", "unknown").upper()
        if status in ("COMPLETED", "COMPLETE", "DONE", "COMPLETED_WITH_ERRORS"):
            print(f" {status.lower()}")
            return r
        if status in ("FAILED", "ERROR"):
            print(f" FAILED: {r}")
            raise RuntimeError(f"Batch failed: {r}")
        print(".", end="", flush=True)
        time.sleep(poll)
    raise TimeoutError(f"Batch {batch_id} did not complete in {timeout}s")


WEB_SCRAPER_FEATURES = [
    "intfloat__multilingual_e5_large_instruct",  # text (1024D)
]


def create_retriever(s, base_url, collection_id, *, name=None,
                     feature_uris=None):
    uris = feature_uris or WEB_SCRAPER_FEATURES
    stages = []
    for uri in uris:
        stages.append(
            {
                "stage_name": uri,
                "stage_type": "filter",
                "config": {
                    "stage_id": "feature_search",
                    "parameters": {
                        "searches": [
                            {
                                "feature_uri": uri,
                                "query": {
                                    "input_mode": "text",
                                    "text": "{{INPUT.query}}",
                                },
                                "top_k": 20,
                            }
                        ],
                        "final_top_k": 20,
                    },
                },
            }
        )
    body = {
        "retriever_name": name or f"pixelrag-search-{uuid.uuid4().hex[:6]}",
        "stages": stages,
        "collection_ids": [collection_id],
        "input_schema": {"query": {"type": "text", "required": True}},
    }
    if len(stages) > 1:
        body["fusion"] = {"type": "rrf"}
    return _post(s, f"{base_url}/v1/retrievers", body)


def execute_search(s, base_url, retriever_id, query):
    return _post(
        s,
        f"{base_url}/v1/retrievers/{retriever_id}/execute",
        {"inputs": {"query": query}},
    )


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="PixelRAG-style visual web search via Mixpeek"
    )
    parser.add_argument("--url", required=True, help="Seed URL to crawl")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--api-key", help="Mixpeek API key (or MIXPEEK_API_KEY env)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--crawl-mode", default="deterministic",
                        choices=["deterministic", "semantic"])
    parser.add_argument("--crawl-goal", help="Goal for semantic crawl mode")
    parser.add_argument("--render-strategy", default="auto")
    parser.add_argument("--chunk-strategy", default="paragraphs")
    args = parser.parse_args()

    import os

    api_key = args.api_key or os.environ.get("MIXPEEK_API_KEY")
    if not api_key:
        print("Error: set MIXPEEK_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    s = _session(api_key)
    base = args.base_url.rstrip("/")

    print("=" * 60)
    print("  PixelRAG → Mixpeek Managed Search")
    print("=" * 60)
    print(f"  URL:        {args.url}")
    print(f"  Query:      {args.query}")
    print(f"  Crawl:      depth={args.max_depth}, max_pages={args.max_pages}")
    print()

    # 1. Namespace (auto-expires)
    ns = create_namespace(s, base)
    ns_id = ns["namespace_id"]
    print(f"  Namespace:  {ns_id} (TTL {TTL_HOURS}h)")

    # 2. Bucket + upload seed URL
    bucket = create_bucket(s, base, ns_id)
    bucket_id = bucket["bucket_id"]
    obj = upload_url(s, base, bucket_id, args.url)
    obj_id = obj.get("object_id") or obj.get("id")
    print(f"  Bucket:     {bucket_id}")
    print(f"  Object:     {obj_id}")

    # 3. Collection (web_scraper extractor)
    col = create_collection(
        s,
        base,
        bucket_id,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        crawl_mode=args.crawl_mode,
        crawl_goal=args.crawl_goal,
        render_strategy=args.render_strategy,
        chunk_strategy=args.chunk_strategy,
    )
    col_id = col["collection_id"]
    print(f"  Collection: {col_id}")

    # 4. Batch → wait
    batch = submit_batch(s, base, bucket_id, col_id, obj_id)
    batch_id = batch.get("batch_id")
    wait_for_batch(s, base, bucket_id, batch_id)

    # 5. Retriever
    retriever = create_retriever(s, base, col_id)
    ret_id = retriever["retriever_id"]
    print(f"  Retriever:  {ret_id}")

    # 6. Search
    results = execute_search(s, base, ret_id, args.query)

    print()
    print("=" * 60)
    print("  Search Results")
    print("=" * 60)

    docs = results.get("documents") or results.get("results") or []
    if not docs:
        print("  (no results)")
    else:
        for i, doc in enumerate(docs[:10]):
            score = doc.get("score", 0)
            label = ""
            for key in ("page_url", "url", "source", "title"):
                if doc.get(key):
                    label = doc[key]
                    break
            if not label:
                content = doc.get("content", "")
                label = (content[:100] + "…") if len(content) > 100 else content
            if not label:
                label = doc.get("document_id", "?")
            print(f"  {i + 1}. {score:.3f}  {label}")

    print()
    print(f"  Namespace expires in {TTL_HOURS}h — no cleanup needed.")
    print(f"  To keep it, remove the TTL via the Mixpeek dashboard.")


if __name__ == "__main__":
    main()
