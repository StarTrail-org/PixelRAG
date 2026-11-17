#!/usr/bin/env python3
"""v2: Generate reasoning traces WITH the actual image in context.

v1 sent only Q+A → GPT hallucinated "look at byline" etc. without knowing if it
was actually visible. v2 sends the screenshot image so the reasoning describes
real visible elements.

Uses detail=low on image (85 tokens) to keep cost ~$3 for 30k calls with mini.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

PROMPT = """Question: {query}
Correct answer: {answer}

Look at this Wikipedia screenshot. In 2-3 sentences, describe concretely what elements in THIS image (section title, infobox row, table cell, byline, date line, etc.) a reader would scan to find the answer. Reference the actual layout and named elements visible in the screenshot. No preamble. Output ONLY the reasoning."""


def encode_image(path: str, max_bytes: int = 4_000_000) -> str | None:
    """Return base64 data-URL for the image; return None if missing/too-big."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) > max_bytes:
            # For large images, re-encode smaller via PIL to stay under limit
            from PIL import Image
            import io

            Image.MAX_IMAGE_PIXELS = 300_000_000
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Shrink longest side to 1024 if larger
            w, h = img.size
            m = max(w, h)
            if m > 1024:
                s = 1024 / m
                img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"  WARN: encode {path}: {e}", file=sys.stderr)
        return None


def process_one(
    client: OpenAI, model: str, ex: dict[str, str], image_root: str
) -> dict[str, str | None]:
    img_path = os.path.join(image_root, ex["chunk_path"])
    img_url = encode_image(img_path) if os.path.exists(img_path) else None
    if img_url is None:
        return {**ex, "reasoning": None, "_error": "no_image"}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(
                                query=ex["query"], answer=ex["answer"]
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url, "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )
        reasoning = resp.choices[0].message.content.strip()
        return {**ex, "reasoning": reasoning}
    except Exception as e:
        return {**ex, "reasoning": None, "_error": str(e)[:200]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--image-root",
        required=True,
        help="Dataset root containing the images/ subtree that chunk_path points into",
    )
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.input) as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} examples")

    if args.n > 0 and args.n < len(data):
        import random

        rng = random.Random(args.seed)
        data = rng.sample(data, args.n)
        print(f"Sampled {len(data)} examples (seed={args.seed})")

    cached = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                if r.get("reasoning"):
                    cached[(r["query"], r["chunk_path"])] = r
        print(f"Found {len(cached)} cached results")

    todo = [ex for ex in data if (ex["query"], ex["chunk_path"]) not in cached]
    print(f"Generating {len(todo)} new image-aware traces with {args.model}")
    if not todo:
        return

    client = OpenAI()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "a", buffering=1)

    fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(process_one, client, args.model, ex, args.image_root)
            for ex in todo
        ]
        for fut in tqdm(
            as_completed(futures), total=len(todo), desc="GPT vision trace"
        ):
            r = fut.result()
            if r.get("reasoning") is None:
                fail += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    fout.close()

    dur = time.time() - t0
    print(f"\nDone in {dur:.1f}s. Failed: {fail}/{len(todo)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
