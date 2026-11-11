#!/usr/bin/env python3
"""
Post-filter: use GPT-4o (or GPT-4o-mini) to classify whether generated queries
are self-contained — BATCH mode (50 queries per API call).

A query is self-contained if it can be understood without knowing which Wikipedia
page or screenshot it came from. Drops queries with vague references ("the team"),
document-layout references ("in the table"), or missing context (no year for a
role that has had many holders).

Usage:
  OPENAI_API_KEY=sk-... python filter_self_contained.py \
      --input raw_query_pairs.jsonl \
      --output filtered_query_pairs.jsonl

  # Test first 100:
  OPENAI_API_KEY=sk-... python filter_self_contained.py \
      --input raw_query_pairs.jsonl \
      --output out.jsonl \
      --test-first 100

Cost estimate (gpt-4o, batch=50):
  72K records → ~1.4K API calls → ~$8 total, ~8 min

Ported from Vis-RAG/agent/scripts/contrastive/filter_self_contained.py
"""

import argparse
import asyncio
import json
import os
import re
import time

import openai

MODEL_PRICING = {
    "gpt-4o-mini": (0.15 / 1e6, 0.60 / 1e6),
    "gpt-4o": (2.50 / 1e6, 10.0 / 1e6),
    "gpt-4.1": (2.00 / 1e6, 8.00 / 1e6),
}

BATCH_SIZE = 50
MAX_CONCURRENT = 50

FILTER_PROMPT = """\
For each numbered question, answer YES (self-contained) or NO (not self-contained).

A question is NOT self-contained (NO) if it requires knowing a specific Wikipedia page, table, or screenshot to understand WHAT is being asked. Specifically answer NO when:

1. The subject is a vague pronoun or generic noun without a proper name:
   NO: "What was the final score of the basketball game between THE TEAM and Marquette?"  ("the team" unnamed)
   NO: "Who directed the episode of THE TELEVISION SERIES titled 'X'?"  ("the television series" unnamed)
   NO: "In what year did THE SUBJECT OF THE ARTICLE move to Tokyo?"  ("the subject" unnamed)
   NO: "What is the running time of THE FILM DESCRIBED IN THE TEXT?"  (layout reference)

2. The question explicitly references document structure:
   NO: "Which item IS LISTED IN THE TABLE as X?"
   NO: "What is shown IN THE INFOBOX?"
   NO: "According to THE PROVIDED TABLE, which..."

3. A role/position question where no year or identifying event is given and the role has had many holders:
   NO: "Who was THE CAPTAIN of HMS Defence?"  (no year, hundreds of captains over centuries)

4. A geographic entity refers only to a category without naming which one:
   NO: "On what date did THE GOODS YARD at the London and North Eastern Railway station close?"  (LNER had hundreds of stations — which one?)

Answer YES if all the key entities (people, places, works, teams, events) are explicitly named, even if the names are obscure. Proper names are always fine.
   YES: "Who did Sandefjord Fotball hire as manager after firing Arne Sandstø?"
   YES: "How many consonants does the Pesisir language have?"
   YES: "What 'fresh' rating did the film Our Man in Havana receive on Rotten Tomatoes?"
   YES: "In what city were the 2025 Special Olympics World Winter Games held?"
   YES: "Who did Émile Derlin Zinsou serve as assistant to in 1945?"

Output exactly one line per question, using the question number: "1: YES" or "1: NO"

Questions:
{questions}"""


def parse_batch_response(text: str, n: int) -> list[bool]:
    results = [True] * n
    if not text:
        return results
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)[:.)\s]+(?:N:\s*)?(YES|NO)", line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            verdict = m.group(2).upper()
            if 0 <= idx < n:
                results[idx] = verdict == "YES"
    return results


async def classify_batch(
    client: openai.AsyncOpenAI,
    records: list[dict],
    semaphore: asyncio.Semaphore,
    token_counter: dict,
    model: str,
) -> list[tuple[dict, bool]]:
    queries = [r["query"] for r in records]
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
    prompt = FILTER_PROMPT.format(questions=numbered)

    async with semaphore:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=BATCH_SIZE * 8,
                )
                text = resp.choices[0].message.content or ""
                token_counter["input"] += resp.usage.prompt_tokens
                token_counter["output"] += resp.usage.completion_tokens
                token_counter["calls"] += 1

                verdicts = parse_batch_response(text, len(records))
                return list(zip(records, verdicts))

            except openai.RateLimitError:
                wait = 20 + attempt * 10
                print(f"  Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Error (attempt {attempt + 1}): {str(e)[:100]}")
                await asyncio.sleep(3)

    return [(r, True) for r in records]


async def main():
    parser = argparse.ArgumentParser(
        description="Filter non-self-contained queries using LLM classification"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL with generated query pairs"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL with only self-contained queries"
    )
    parser.add_argument(
        "--test-first",
        type=int,
        default=None,
        help="Only process first N records (dry run)",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model for classification"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    data = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass

    if args.test_first:
        data = data[: args.test_first]

    batches = [
        data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)
    ]

    print(f"Loaded {len(data)} records → {len(batches)} batches of {args.batch_size}")
    print(f"Model: {args.model}  |  Concurrency: {args.concurrency}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    semaphore = asyncio.Semaphore(args.concurrency)
    price_in, price_out = MODEL_PRICING.get(args.model, (2.50 / 1e6, 10.0 / 1e6))
    token_counter = {"input": 0, "output": 0, "calls": 0}

    t0 = time.time()
    tasks = [
        classify_batch(client, b, semaphore, token_counter, args.model) for b in batches
    ]
    batch_results = await asyncio.gather(*tasks)

    all_results = [item for batch in batch_results for item in batch]
    kept = [r for r, ok in all_results if ok]
    dropped = [r for r, ok in all_results if not ok]

    elapsed = time.time() - t0
    cost = token_counter["input"] * price_in + token_counter["output"] * price_out

    print(f"\n{'=' * 60}")
    print(f"Total:   {len(data)}")
    print(f"Kept:    {len(kept)}  ({len(kept) / len(data) * 100:.1f}%)")
    print(f"Dropped: {len(dropped)}  ({len(dropped) / len(data) * 100:.1f}%)")
    print(
        f"Calls:   {token_counter['calls']}  |  Time: {elapsed:.1f}s  |  Cost: ${cost:.4f}"
    )
    print(f"{'=' * 60}")

    print(f"\n--- Sample dropped queries ({min(30, len(dropped))}) ---")
    for r in dropped[:30]:
        print(f"  [{r.get('source_type', '?'):7s}] {r['query']}")

    if args.test_first:
        print("\n[TEST MODE] Not writing output.")
        return

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept)} records to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
