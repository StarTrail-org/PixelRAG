#!/usr/bin/env python3
"""Clean query-only training rows toward a SimpleQA-like style.

This script reads one or more JSONL files that contain `query` fields and uses
Gemini to judge each query on two axes:
1. Naturalness: does it sound like a real user question?
2. SimpleQA style fit: does it resemble the style of SimpleQA factoid prompts?

The model only sees the query text. It does not inspect images or answers.

Outputs:
- cleaned JSONL with original rows preserved
- review JSONL with model scores / keep decisions
- summary JSON with before/after stats and token usage
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

DEFAULT_INPUT_GLOB = (
    "training/data/lite-query-v2-full-filtered-hn-v2-chunks/chunk_*/filtered_hn.jsonl"
)
DEFAULT_SIMPLEQA_PATH = (
    "/home/user/wiki-screenshot/eval/simpleqa_query_image_pairs.json"
)

MODEL_PRICING = {
    "gemini-2.0-flash-001": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-2.0-flash": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-3.1-flash": {"input_per_m": 0.25, "output_per_m": 1.50},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob pattern for input JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for cleaned rows.",
    )
    parser.add_argument(
        "--reviews-output",
        default=None,
        help="Optional JSONL path for model review decisions.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON path for summary stats.",
    )
    parser.add_argument(
        "--simpleqa-path",
        default=DEFAULT_SIMPLEQA_PATH,
        help="Path to SimpleQA pair JSON used for style references.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50000,
        help="Approximate number of rows to keep.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Queries per Gemini request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent Gemini requests.",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=12,
        help="Number of SimpleQA examples shown in the prompt.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-001",
        help="Gemini model name. Use an accessible model in your project.",
    )
    parser.add_argument("--gemini-project", default="wise-coyote-478119-h0")
    parser.add_argument("--gemini-location", default="global")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-naturalness",
        type=int,
        default=4,
        help="Minimum Gemini naturalness score for direct keep.",
    )
    parser.add_argument(
        "--min-style-fit",
        type=int,
        default=4,
        help="Minimum Gemini SimpleQA style score for direct keep.",
    )
    parser.add_argument(
        "--dedupe-query",
        action="store_true",
        help="Keep only the highest-scoring row per normalized query.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing review file and skip already-scored row ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on total input rows for smoke tests.",
    )
    return parser.parse_args()


def init_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "calls": 0,
    }


def build_client(args: argparse.Namespace) -> dict:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", args.gemini_project)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", args.gemini_location)

    from google import genai
    from google.genai.types import HttpOptions

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    return {
        "client": client,
        "usage": init_usage(),
        "usage_lock": threading.Lock(),
    }


def update_usage(
    client_ctx: dict, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    with client_ctx["usage_lock"]:
        client_ctx["usage"]["prompt_tokens"] += int(prompt_tokens or 0)
        client_ctx["usage"]["completion_tokens"] += int(completion_tokens or 0)
        client_ctx["usage"]["calls"] += 1


def parse_json_from_text(text: str):
    text = text.strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if not match:
        raise ValueError("no JSON object or array found")
    return json.loads(match.group(1))


def call_gemini_json(client_ctx: dict, model: str, prompt: str, max_retries: int):
    from google.genai.types import GenerateContentConfig

    config = GenerateContentConfig(
        temperature=0,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client_ctx["client"].models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                update_usage(
                    client_ctx,
                    prompt_tokens=getattr(usage, "prompt_token_count", 0),
                    completion_tokens=getattr(usage, "candidates_token_count", 0),
                )
            text = getattr(resp, "text", "") or ""
            return parse_json_from_text(text)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("unreachable")


def iter_rows(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_input_rows(args: argparse.Namespace) -> list[dict]:
    rows = []
    matched = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not matched:
        raise FileNotFoundError(f"No files matched --input-glob={args.input_glob}")

    row_id = 0
    for path in matched:
        for line_no, payload in iter_rows(path):
            query = payload.get("query") or payload.get("question")
            if not isinstance(query, str) or not query.strip():
                continue
            rows.append(
                {
                    "row_id": row_id,
                    "source_file": str(path),
                    "source_line": line_no,
                    "query": query.strip(),
                    "payload": payload,
                }
            )
            row_id += 1
            if args.limit > 0 and len(rows) >= args.limit:
                return rows
    return rows


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def question_start_bucket(query: str) -> str:
    q = query.strip().lower()
    for prefix in [
        "what",
        "who",
        "which",
        "when",
        "where",
        "why",
        "how",
        "in which",
        "in what",
        "on what",
        "what is",
        "what was",
    ]:
        if q.startswith(prefix):
            return prefix
    parts = re.findall(r"[a-z0-9']+", q)
    return parts[0] if parts else "<empty>"


def load_simpleqa_references(path: Path, few_shot_count: int, seed: int) -> list[dict]:
    with path.open() as f:
        data = json.load(f)

    by_bucket = {}
    for item in data:
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        bucket = question_start_bucket(question)
        by_bucket.setdefault(bucket, []).append(
            {
                "question": question.strip(),
                "answer": (answer or "").strip(),
                "topic": item.get("topic") or "Unknown",
            }
        )

    rng = random.Random(seed)
    for values in by_bucket.values():
        rng.shuffle(values)

    ordered_buckets = sorted(by_bucket, key=lambda key: (-len(by_bucket[key]), key))
    refs = []
    while len(refs) < few_shot_count and ordered_buckets:
        next_round = []
        for bucket in ordered_buckets:
            values = by_bucket[bucket]
            if values:
                refs.append(values.pop())
            if values:
                next_round.append(bucket)
            if len(refs) >= few_shot_count:
                break
        ordered_buckets = next_round
    return refs


def build_prompt(reference_examples: list[dict], batch: list[dict]) -> str:
    ref_lines = []
    for idx, ref in enumerate(reference_examples, 1):
        ref_lines.append(
            f"{idx}. topic={ref['topic']}\n"
            f"   question={json.dumps(ref['question'], ensure_ascii=False)}\n"
            f"   answer={json.dumps(ref['answer'], ensure_ascii=False)}"
        )

    candidate_json = json.dumps(
        [{"id": row["row_id"], "query": row["query"]} for row in batch],
        ensure_ascii=False,
        indent=2,
    )

    return f"""You are cleaning a synthetic screenshot-retrieval training set.

Goal: keep only queries that read naturally and resemble SimpleQA-style factoid questions.
Judge ONLY the query text. Do not assume access to images, answers, or metadata.

High-quality queries usually:
- sound like something a real user would ask
- are single-hop factoid questions with short answers
- include natural disambiguating context when useful (time, role, location, quoted title, relation)
- feel similar to the reference questions below

Reject queries that are:
- awkward, templatic, or annotation-like
- unnaturally phrased, especially stiff starts like "In what..." or "At which..." when a plain wording would be more natural
- keywordy, malformed, or obviously synthetic
- broad explanatory prompts, opinion questions, yes/no questions, or multi-hop questions

Reference SimpleQA-style examples:
{chr(10).join(ref_lines)}

Now score these candidate queries.

Return ONLY a JSON array. One object per candidate with exactly these keys:
- id: integer
- naturalness: integer 1-5
- simpleqa_style_fit: integer 1-5
- keep: boolean
- reason: short string (<=12 words)

Scoring guide:
- 5 = excellent
- 4 = good / clearly keepable
- 3 = borderline
- 2 = weak
- 1 = poor

Candidates:
{candidate_json}
"""


def sanitize_decision(raw: dict, row_id: int) -> dict:
    naturalness = raw.get("naturalness", 0)
    style_fit = raw.get("simpleqa_style_fit", 0)
    keep = raw.get("keep", False)
    reason = raw.get("reason", "")

    try:
        naturalness = int(naturalness)
    except Exception:
        naturalness = 0
    try:
        style_fit = int(style_fit)
    except Exception:
        style_fit = 0
    if isinstance(keep, str):
        keep = keep.strip().lower() in {"true", "yes", "1", "keep"}

    return {
        "id": int(row_id),
        "naturalness": max(0, min(5, naturalness)),
        "simpleqa_style_fit": max(0, min(5, style_fit)),
        "keep": bool(keep),
        "reason": str(reason).strip()[:200],
    }


def score_batch(
    client_ctx: dict,
    args: argparse.Namespace,
    references: list[dict],
    batch: list[dict],
) -> list[dict]:
    prompt = build_prompt(references, batch)
    parsed = call_gemini_json(client_ctx, args.model, prompt, args.max_retries)
    # TRY004 asks for TypeError here. This validates a model's response
    # payload rather than a caller's argument, so ValueError is the right
    # contract and callers already catch it.
    if not isinstance(parsed, list):
        raise ValueError("Gemini response is not a JSON array")  # noqa: TRY004

    by_id = {}
    for item in parsed:
        if not isinstance(item, dict) or "id" not in item:
            continue
        by_id[int(item["id"])] = sanitize_decision(item, item["id"])

    decisions = []
    for row in batch:
        decision = by_id.get(row["row_id"])
        if decision is None:
            decision = {
                "id": row["row_id"],
                "naturalness": 0,
                "simpleqa_style_fit": 0,
                "keep": False,
                "reason": "missing_from_model_output",
            }
        decisions.append(decision)
    return decisions


def load_existing_reviews(path: Path) -> dict[int, dict]:
    existing = {}
    if not path.exists():
        return existing
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            existing[int(item["row_id"])] = item
    return existing


def review_rows(
    rows: list[dict],
    args: argparse.Namespace,
    client_ctx: dict,
    references: list[dict],
    reviews_path: Path,
) -> dict[int, dict]:
    existing = load_existing_reviews(reviews_path) if args.resume else {}
    pending = [row for row in rows if row["row_id"] not in existing]
    total = len(rows)

    if pending:
        reviews_path.parent.mkdir(parents=True, exist_ok=True)
        with reviews_path.open("a") as reviews_file:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {}
                pending_batches = [
                    pending[start : start + args.batch_size]
                    for start in range(0, len(pending), args.batch_size)
                ]
                next_batch_idx = 0

                while (
                    next_batch_idx < len(pending_batches)
                    and len(futures) < args.concurrency
                ):
                    batch = pending_batches[next_batch_idx]
                    future = executor.submit(
                        score_batch, client_ctx, args, references, batch
                    )
                    futures[future] = batch
                    next_batch_idx += 1

                completed = 0
                while futures:
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        batch = futures.pop(future)
                        decisions = future.result()
                        for row, decision in zip(batch, decisions):
                            review = {
                                "row_id": row["row_id"],
                                "query": row["query"],
                                "source_file": row["source_file"],
                                "source_line": row["source_line"],
                                **decision,
                            }
                            existing[row["row_id"]] = review
                            reviews_file.write(
                                json.dumps(review, ensure_ascii=False) + "\n"
                            )
                        reviews_file.flush()
                        completed += len(batch)
                        print(
                            f"Reviewed {len(existing)}/{total} rows "
                            f"(new {completed}/{len(pending)})",
                            flush=True,
                        )
                        if next_batch_idx < len(pending_batches):
                            next_batch = pending_batches[next_batch_idx]
                            next_future = executor.submit(
                                score_batch, client_ctx, args, references, next_batch
                            )
                            futures[next_future] = next_batch
                            next_batch_idx += 1
    return existing


def candidate_priority(review: dict) -> tuple:
    return (
        int(review["keep"]),
        int(review["naturalness"]) + int(review["simpleqa_style_fit"]),
        int(review["simpleqa_style_fit"]),
        int(review["naturalness"]),
        -int(review["row_id"]),
    )


def select_rows(
    rows: list[dict], reviews: dict[int, dict], args: argparse.Namespace
) -> list[dict]:
    candidates = []
    for row in rows:
        review = reviews.get(row["row_id"])
        if not review:
            continue
        direct_keep = (
            review["keep"]
            and review["naturalness"] >= args.min_naturalness
            and review["simpleqa_style_fit"] >= args.min_style_fit
        )
        if direct_keep:
            candidates.append((row, review))

    if args.dedupe_query:
        by_query = {}
        for row, review in candidates:
            key = normalize_query(row["query"])
            current = by_query.get(key)
            if current is None or candidate_priority(review) > candidate_priority(
                current[1]
            ):
                by_query[key] = (row, review)
        candidates = list(by_query.values())

    candidates.sort(key=lambda item: candidate_priority(item[1]), reverse=True)

    if args.target_count > 0 and len(candidates) > args.target_count:
        candidates = candidates[: args.target_count]

    return [row for row, _ in candidates]


def word_count(query: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", query))


def compute_query_stats(queries: list[str]) -> dict:
    if not queries:
        return {
            "count": 0,
            "avg_words": 0.0,
            "avg_chars": 0.0,
            "top_starts": [],
            "has_quote_pct": 0.0,
            "has_year_pct": 0.0,
        }

    starts = Counter(question_start_bucket(query) for query in queries)
    word_counts = [word_count(query) for query in queries]
    char_counts = [len(query) for query in queries]
    has_quote = sum('"' in query or "'" in query for query in queries)
    has_year = sum(
        bool(re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", query)) for query in queries
    )
    return {
        "count": len(queries),
        "avg_words": round(sum(word_counts) / len(word_counts), 2),
        "avg_chars": round(sum(char_counts) / len(char_counts), 2),
        "top_starts": starts.most_common(12),
        "has_quote_pct": round(100 * has_quote / len(queries), 2),
        "has_year_pct": round(100 * has_year / len(queries), 2),
    }


def estimated_cost_usd(model: str, usage: dict) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    in_cost = usage["prompt_tokens"] / 1_000_000 * pricing["input_per_m"]
    out_cost = usage["completion_tokens"] / 1_000_000 * pricing["output_per_m"]
    return round(in_cost + out_cost, 6)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row["payload"], ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    reviews_path = (
        Path(args.reviews_output)
        if args.reviews_output
        else output_path.with_suffix(".reviews.jsonl")
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )

    random.seed(args.seed)

    rows = load_input_rows(args)
    print(f"Loaded {len(rows)} rows from {args.input_glob}", flush=True)

    references = load_simpleqa_references(
        Path(args.simpleqa_path), args.few_shot_count, args.seed
    )
    print(f"Loaded {len(references)} SimpleQA reference examples", flush=True)

    client_ctx = build_client(args)
    reviews = review_rows(rows, args, client_ctx, references, reviews_path)
    selected = select_rows(rows, reviews, args)
    write_jsonl(output_path, selected)

    input_queries = [row["query"] for row in rows]
    output_queries = [row["query"] for row in selected]
    summary = {
        "model": args.model,
        "input_glob": args.input_glob,
        "simpleqa_path": args.simpleqa_path,
        "total_input_rows": len(rows),
        "reviewed_rows": len(reviews),
        "selected_rows": len(selected),
        "target_count": args.target_count,
        "dedupe_query": args.dedupe_query,
        "min_naturalness": args.min_naturalness,
        "min_style_fit": args.min_style_fit,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "query_stats_before": compute_query_stats(input_queries),
        "query_stats_after": compute_query_stats(output_queries),
        "usage": {
            **client_ctx["usage"],
            "estimated_cost_usd": estimated_cost_usd(args.model, client_ctx["usage"]),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "reviews_output": str(reviews_path),
                "summary_output": str(summary_path),
                "selected_rows": len(selected),
                "estimated_cost_usd": summary["usage"]["estimated_cost_usd"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
