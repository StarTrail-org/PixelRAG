#!/usr/bin/env python3
"""Generate reasoning traces for SFT think training.

For each (query, answer) in train_hn_with_answer.jsonl, ask GPT-4.1-mini to
synthesize a short reasoning trace that could plausibly lead to the answer
given a screenshot. We send text-only (no image) since sending 100k images is
too expensive — the trace represents "plausible thought process," not ground
truth.

Output: JSONL with {query, chunk_path, answer, reasoning} fields.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

PROMPT = """Given this fact-lookup question from a Wikipedia screenshot:

Question: {query}
Correct answer: {answer}

Write a brief reasoning trace (2-3 sentences) showing how someone would find this answer by examining the screenshot. Mention what specific text/detail they would look for. Be natural and concise. No preamble. Output ONLY the reasoning, nothing else."""


def process_one(
    client: OpenAI, model: str, ex: dict[str, str]
) -> dict[str, str | None]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(query=ex["query"], answer=ex["answer"]),
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
    p.add_argument("--n", type=int, default=20000, help="Sample N examples (0=all)")
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

    # Check what's already cached (resume support)
    cached = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                if r.get("reasoning"):
                    cached[(r["query"], r["chunk_path"])] = r
        print(f"Found {len(cached)} cached results")

    todo = [ex for ex in data if (ex["query"], ex["chunk_path"]) not in cached]
    print(f"Generating {len(todo)} new reasoning traces with {args.model}")
    if not todo:
        print("Nothing to do.")
        return

    client = OpenAI()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode
    fout = open(args.output, "a", buffering=1)

    fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process_one, client, args.model, ex) for ex in todo]
        for fut in tqdm(as_completed(futures), total=len(todo), desc="GPT trace"):
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
