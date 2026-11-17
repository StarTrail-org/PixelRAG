"""Filter for the hardest training examples: answers buried in tables or dense passages.

Two target categories:
  TABLE_DEEP — answer requires locating a specific cell in a multi-row table
    e.g., "Who won the X award in 2014?", "What was the score in match Y?"
    These train the model to distinguish between table chunks vs overview chunks.

  PASSAGE_DENSE — answer is a small detail mentioned once in a long prose paragraph
    e.g., "What nickname does X use for Y?", "Who was demoted after the battle?"
    These train the model to match queries to specific narrative sections.

Excludes:
  - Answers from infobox/sidebar (too easy, always chunk_0)
  - Answers from bullet-point lists (semi-structured, less valuable)
  - General overview questions
  - Simple factoid questions answerable from intro

Uses GPT-4.1-mini for classification with strict prompt.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM = """You classify queries for a visual document retrieval training dataset.

We want ONLY the hardest queries — ones where finding the answer requires scanning deep into a Wikipedia article's tables or paragraphs.

Classify each query into exactly ONE category:

**T** (TABLE_DEEP) — answer is in a specific cell/row of a data table:
- Match statistics: scores, goals, fouls, attendance numbers
- Award recipients by year: "Who won X in 2014?"
- Tournament/competition results: rankings, times, distances
- Census/demographic tables: population by year
- Episode/season tables: air dates, viewer counts, guest stars
- Transfer tables: fees, dates, clubs
- Election results: vote counts, percentages
- Chemical/physical property tables: boiling point, density
- Financial tables: revenue, budget, salary figures
NOT infobox questions (birth date, capital, language — those are too easy)

**P** (PASSAGE_DENSE) — answer is a small detail buried in a prose paragraph:
- A name mentioned once in a narrative: "Who replaced X after the incident?"
- A specific detail in a story: "What nickname does X call Y?"
- A fact embedded in historical narrative: "Who was sent to negotiate the treaty?"
- A detail from a review/criticism section: "According to X, what was the main flaw?"
- A cause/reason explained in text: "Why was X expelled from the party?"
- A quote or claim attributed to someone in prose
NOT questions where the answer is the main topic of a paragraph

**X** (EXCLUDE) — not hard enough:
- Infobox/sidebar questions (birth, death, nationality, founded, capital)
- Main topic of the article (answerable from title + first sentence)
- Simple "What is X?" overview questions
- Bullet-point list items
- Questions about the primary subject's basic attributes
- Anything answerable from the article's introduction paragraph

Be STRICT with T and P — only mark T/P if the answer genuinely requires scanning through dense content. When in doubt, mark X.

Respond with ONLY a JSON array of single characters: "T", "P", or "X"."""

USER_TPL = """Classify {n} queries. JSON array of "T"/"P"/"X" only.

{block}"""


async def classify_batch(client, queries, batch_idx, sem):
    async with sem:
        block = "\n".join(f"[{i}] {q}" for i, q in enumerate(queries))
        for attempt in range(5):
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": USER_TPL.format(n=len(queries), block=block),
                            },
                        ],
                        "temperature": 0,
                        "max_tokens": 400,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                labels = json.loads(content)
                if isinstance(labels, list):
                    while len(labels) < len(queries):
                        labels.append("X")
                    return batch_idx, [
                        str(x).upper().strip('"')[:1] for x in labels[: len(queries)]
                    ]
                return batch_idx, ["X"] * len(queries)
            except Exception as e:
                if attempt == 4:
                    logger.warning(f"Batch {batch_idx}: {e}")
                    return batch_idx, ["X"] * len(queries)
                await asyncio.sleep(2**attempt)


def parse_pos(path):
    m = re.search(r"chunk_(\d+)_(\d+)\.png", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://us.api.openai.com/v1")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    logger.info(f"Loading {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    N = len(records)
    logger.info(f"Loaded {N} records")

    # Pre-filter: skip c0 chunk records (likely infobox/intro answers)
    # But keep c0 if query has strong table/passage signals
    TABLE_SIGNAL = re.compile(
        r"\b(score|goals|fouls|runs|wickets|attendance|votes|percentage|population|"
        r"strike rate|batting|bowling|transfer fee|revenue|budget|salary|"
        r"boiling point|melting point|molar|atomic mass|density|"
        r"award|prize|medal|trophy|championship).*\d|"
        r"\d{4}.*\b(award|prize|medal|trophy)\b|"
        r"\b(episode|season)\s+\d.*\b(air|viewer|rating|guest)\b|"
        r"\bhow many (goals|points|runs|votes|fouls|wickets|spectators|viewers)\b",
        re.IGNORECASE,
    )
    PASSAGE_SIGNAL = re.compile(
        r"\b(nickname|called|referred to|according to|why did|why was|"
        r"what caused|what led to|what prompted|who replaced|who succeeded|"
        r"who was sent|who negotiated|what role did|what position did|"
        r"what was the (reason|cause|result|outcome|consequence)|"
        r"after the (battle|war|incident|death|resignation|defeat)|"
        r"during the (meeting|ceremony|trial|debate))\b",
        re.IGNORECASE,
    )

    to_classify_idx = []
    auto_exclude = 0
    for i, r in enumerate(records):
        page, chunk = parse_pos(r["chunk_path"])
        q = r["query"]
        # Auto-exclude: c0 without any signal
        if page == 0 and chunk == 0:
            if TABLE_SIGNAL.search(q) or PASSAGE_SIGNAL.search(q):
                to_classify_idx.append(i)
            else:
                auto_exclude += 1
        else:
            to_classify_idx.append(i)

    logger.info(f"Auto-excluded c0 without signals: {auto_exclude}")
    logger.info(f"To classify: {len(to_classify_idx)}")

    # Batch for LLM
    classify_pairs = [(idx, records[idx]["query"]) for idx in to_classify_idx]
    batches = []
    for i in range(0, len(classify_pairs), args.batch_size):
        batches.append(classify_pairs[i : i + args.batch_size])
    logger.info(f"{len(batches)} batches")

    labels_map = {}  # idx -> label
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    ) as client:
        tasks = [
            classify_batch(client, [q for _, q in b], bi, sem)
            for bi, b in enumerate(batches)
        ]
        done = 0
        t0 = time.time()
        for coro in asyncio.as_completed(tasks):
            bi, labels = await coro
            for (orig_idx, _), lbl in zip(batches[bi], labels):
                labels_map[orig_idx] = lbl
            done += 1
            if done % 100 == 0 or done == len(tasks):
                t_count = sum(1 for v in labels_map.values() if v == "T")
                p_count = sum(1 for v in labels_map.values() if v == "P")
                x_count = sum(1 for v in labels_map.values() if v == "X")
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 1
                eta = (len(tasks) - done) / rate
                logger.info(
                    f"{done}/{len(tasks)}: T={t_count} P={p_count} X={x_count} "
                    f"ETA {eta:.0f}s"
                )

    # Collect results
    table_records = []
    passage_records = []
    for idx in to_classify_idx:
        lbl = labels_map.get(idx, "X")
        if lbl == "T":
            table_records.append(records[idx])
        elif lbl == "P":
            passage_records.append(records[idx])

    combined = table_records + passage_records

    logger.info("\n=== RESULTS ===")
    logger.info(f"TABLE_DEEP:    {len(table_records)}")
    logger.info(f"PASSAGE_DENSE: {len(passage_records)}")
    logger.info(f"EXCLUDED:      {N - len(combined)}")
    logger.info(f"COMBINED:      {len(combined)} ({len(combined) / N * 100:.1f}%)")

    # Position breakdown
    from collections import Counter

    for tag, recs in [("TABLE", table_records), ("PASSAGE", passage_records)]:
        pos = Counter()
        for r in recs:
            p, c = parse_pos(r["chunk_path"])
            if p >= 1:
                pos["page1+"] += 1
            elif c <= 1:
                pos["c0-1"] += 1
            elif c <= 3:
                pos["c2-3"] += 1
            else:
                pos["c4+"] += 1
        logger.info(f"  {tag} positions: {dict(sorted(pos.items()))}")

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    for name, recs in [
        ("train_table.jsonl", table_records),
        ("train_passage.jsonl", passage_records),
        ("train_hard.jsonl", combined),
    ]:
        path = os.path.join(args.output_dir, name)
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(recs):,} → {path}")

    # Create split dir with eval/test from original
    split_dir = os.path.join(args.output_dir, "split")
    os.makedirs(split_dir, exist_ok=True)

    import shutil

    src_split = "training/data/natrual_filtered_v2/split"
    shutil.copy2(f"{src_split}/eval_hn.jsonl", f"{split_dir}/eval_hn.jsonl")
    shutil.copy2(f"{src_split}/test_hn.jsonl", f"{split_dir}/test_hn.jsonl")
    with open(f"{split_dir}/train_hn.jsonl", "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Symlink images
    images_src = os.path.realpath(f"{src_split}/images")
    images_dst = f"{split_dir}/images"
    if os.path.exists(images_dst):
        os.remove(images_dst)
    os.symlink(images_src, images_dst)

    logger.info(f"\nReady to use: --data-split-dir {split_dir}")

    # Save samples for visual verification
    import random

    random.seed(42)
    for tag, recs in [("table", table_records), ("passage", passage_records)]:
        sample = random.sample(recs, min(100, len(recs)))
        sf = os.path.join(args.output_dir, f"sample_{tag}_100.jsonl")
        with open(sf, "w") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Sample: {sf}")


if __name__ == "__main__":
    asyncio.run(main())
