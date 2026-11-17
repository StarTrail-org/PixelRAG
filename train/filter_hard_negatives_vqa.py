#!/usr/bin/env python3
"""Filter false-negative retrieval candidates into usable hard negatives.

Input format: JSONL rows containing at least:
- query
- chunk_path
- retrieve_top20

For each example:
1. Scan retrieve_top20 in rank order
2. Skip the positive chunk_path
3. For up to K candidates, ask the VLM to answer the query from that image
4. Judge the answer on the same image
5. If judge == CORRECT -> false negative (FN), skip it
6. If judge == WRONG or CANNOT_ANSWER -> hard negative (HN), keep it

If the positive chunk itself cannot answer the query correctly, the example is
skipped before mining negatives. If fewer than N HNs are found within the first
K non-positive candidates, the example is skipped.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

CHAT_COMPLETIONS_URL = "https://us.api.openai.com/v1/chat/completions"
MODEL_PRICING = {
    # OpenAI pricing is approximate and intended for side-by-side smoke-test estimates.
    "gpt-4.1-mini": {"input_per_m": 0.40, "output_per_m": 1.60},
    # Vertex/Gemini pricing copied from the existing query-pair generation script.
    "gemini-2.0-flash-001": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-2.0-flash": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-2.5-flash-preview-04-17": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gemini-2.5-pro": {"input_per_m": 1.25, "output_per_m": 10.00},
    "gemini-2.5-pro-preview-03-25": {"input_per_m": 1.25, "output_per_m": 10.00},
    "gemini-3.1-pro-preview": {"input_per_m": 1.25, "output_per_m": 10.00},
}

ANSWER_PROMPT_TEMPLATE = """You are looking at {tile_count} screenshot tiles from Wikipedia pages.

Based ONLY on what you can see in these images, answer the following question.
If the answer is not visible in the images, reply "CANNOT_ANSWER".

Question: {question}

Give a short, direct answer (just the answer, no explanation)."""

JUDGE_PROMPT_TEMPLATE = """You are validating a candidate answer against screenshot tiles from Wikipedia pages.

Based ONLY on what you can see in these images, classify the candidate answer to the question as exactly one of:
- CORRECT: the candidate answer is visible in the images and is correct.
- WRONG: the images contain enough information to tell that the candidate answer is wrong.
- CANNOT_ANSWER: the images do not contain enough information to verify the candidate answer.

Question: {question}
Candidate answer: {candidate_answer}

Return exactly one token: CORRECT, WRONG, or CANNOT_ANSWER."""


class MissingImageError(FileNotFoundError):
    """Raised when a referenced image path no longer exists on disk."""


class ApiRequestError(RuntimeError):
    """Raised when VLM API calls fail after retries."""


def init_token_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "calls": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input JSONL with retrieve_top20"
    )
    parser.add_argument("--output", required=True, help="Filtered output JSONL")
    parser.add_argument(
        "--reviews-output", default=None, help="Optional candidate review JSONL"
    )
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many input rows before processing",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max examples to process (0=all)"
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Max non-positive candidates to inspect",
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=2,
        help="Number of HNs to keep per example",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Which VLM provider to use for answer/judge calls.",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Alias for --provider gemini.",
    )
    parser.add_argument("--gemini-project", default="wise-coyote-478119-h0")
    parser.add_argument("--gemini-location", default="global")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of examples to process in parallel.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path, offset: int, limit: int):
    yielded = 0
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            yielded += 1
            if limit > 0 and yielded >= limit:
                break


def encode_image_as_data_url(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "image/png" if ext == "png" else f"image/{ext}"
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise MissingImageError(path) from exc
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def infer_image_mime(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    return "image/png" if ext == "png" else f"image/{ext}"


def encode_image_base64(path: str) -> str:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise MissingImageError(path) from exc
    return base64.b64encode(raw).decode("ascii")


def update_usage(
    client_ctx: dict, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    with client_ctx["usage_lock"]:
        client_ctx["usage"]["prompt_tokens"] += int(prompt_tokens or 0)
        client_ctx["usage"]["completion_tokens"] += int(completion_tokens or 0)
        client_ctx["usage"]["calls"] += 1


def build_vlm_client(args: argparse.Namespace) -> dict:
    provider = "gemini" if args.gemini else args.provider
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return {
            "provider": "openai",
            "api_key": api_key,
            "usage": init_token_usage(),
            "usage_lock": threading.Lock(),
        }

    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", args.gemini_project)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", args.gemini_location)
    from google import genai
    from google.genai.types import HttpOptions

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    return {
        "provider": "gemini",
        "client": client,
        "usage": init_token_usage(),
        "usage_lock": threading.Lock(),
    }


def call_openai_chat_completions(
    client_ctx: dict, model: str, prompt: str, image_path: str, max_retries: int
) -> str:
    headers = {
        "Authorization": f"Bearer {client_ctx['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_as_data_url(image_path)},
                    },
                ],
            }
        ],
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=180
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            update_usage(
                client_ctx,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if attempt == max_retries:
                raise ApiRequestError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("Unreachable")


def call_gemini_generate_content(
    client_ctx: dict, model: str, prompt: str, image_path: str, max_retries: int
) -> str:
    from google.genai.types import GenerateContentConfig

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": infer_image_mime(image_path),
                        "data": encode_image_base64(image_path),
                    }
                },
            ],
        }
    ]
    config = GenerateContentConfig(temperature=0, max_output_tokens=128)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client_ctx["client"].models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                update_usage(
                    client_ctx,
                    prompt_tokens=getattr(usage, "prompt_token_count", 0),
                    completion_tokens=getattr(usage, "candidates_token_count", 0),
                )
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            candidates = getattr(resp, "candidates", None) or []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    raw = getattr(part, "text", None)
                    if raw and not getattr(part, "thought", False):
                        return raw.strip()
            raise ApiRequestError("Gemini returned no text content")
        except MissingImageError:
            raise
        except Exception as exc:
            if attempt == max_retries:
                raise ApiRequestError(f"{type(exc).__name__}: {exc}") from exc
            wait_seconds = min(2**attempt, 20)
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait_seconds = max(wait_seconds, 15)
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable")


def call_vlm(
    client_ctx: dict, model: str, prompt: str, image_path: str, max_retries: int
) -> str:
    if client_ctx["provider"] == "openai":
        return call_openai_chat_completions(
            client_ctx, model, prompt, image_path, max_retries
        )
    return call_gemini_generate_content(
        client_ctx, model, prompt, image_path, max_retries
    )


def answer_question(
    client_ctx: dict, model: str, question: str, image_path: str, max_retries: int
) -> str:
    prompt = ANSWER_PROMPT_TEMPLATE.format(tile_count=1, question=question)
    return call_vlm(client_ctx, model, prompt, image_path, max_retries)


def judge_answer(
    client_ctx: dict,
    model: str,
    question: str,
    image_path: str,
    answer: str,
    max_retries: int,
) -> str:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, candidate_answer=answer)
    verdict = call_vlm(client_ctx, model, prompt, image_path, max_retries)
    verdict = verdict.strip().upper().replace('"', "").replace("`", "")
    compact_verdict = verdict.replace("-", "_").replace(" ", "_")
    for candidate in ("CANNOT_ANSWER", "CORRECT", "WRONG"):
        if candidate in compact_verdict:
            verdict = candidate
            break
    else:
        # Gemini sometimes ignores the "one token" instruction and returns a
        # rationale instead. Treat ambiguous free-form responses as non-CORRECT
        # so we never crash a long-running chunk and never silently keep a FN.
        if any(
            marker in verdict
            for marker in (
                "NOT ENOUGH INFORMATION",
                "CANNOT BE VERIFIED",
                "CANNOT VERIFY",
                "CANNOT DETERMINE",
                "UNABLE TO DETERMINE",
                "NOT VISIBLE",
                "NOT SHOWN",
                "DOES NOT SHOW",
                "DOESN'T SHOW",
            )
        ):
            verdict = "CANNOT_ANSWER"
        else:
            verdict = "WRONG"
    return verdict


def normalize_answer(answer: str) -> str:
    return answer.strip().upper().replace('"', "").replace("`", "")


def init_counts() -> dict:
    return {
        "candidate_verdicts": {"CORRECT": 0, "WRONG": 0, "CANNOT_ANSWER": 0},
        "path_stats": {
            "positive_paths_checked": 0,
            "positive_paths_missing": 0,
            "candidate_paths_checked": 0,
            "candidate_paths_missing": 0,
            "all_paths_checked": 0,
            "all_paths_missing": 0,
        },
        "skip_reasons": {
            "not_enough_hard_negatives": 0,
            "positive_path_missing": 0,
            "positive_not_correct": 0,
            "api_error": 0,
        },
        "examples_with_missing_candidate_paths": 0,
    }


def get_candidates(item: dict, candidate_k: int) -> list[dict]:
    positive_path = item["chunk_path"]
    out = []
    for cand in item.get("retrieve_top20", []):
        if cand["path"] == positive_path:
            continue
        out.append(cand)
        if len(out) >= candidate_k:
            break
    return out


def append_jsonl(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    tmp_path.replace(path)


def path_exists(path: str, cache: dict[str, bool], lock: threading.Lock) -> bool:
    with lock:
        cached = cache.get(path)
    if cached is not None:
        return cached
    exists = Path(path).exists()
    with lock:
        cache[path] = exists
    return exists


def process_example(
    example_index: int,
    item: dict,
    args: argparse.Namespace,
    client_ctx: dict,
    path_cache: dict[str, bool],
    path_cache_lock: threading.Lock,
) -> dict:
    query = item["query"]
    selected_hns = []
    review_rows = []
    counts = init_counts()
    positive_path = item["chunk_path"]

    counts["path_stats"]["positive_paths_checked"] += 1
    counts["path_stats"]["all_paths_checked"] += 1
    if not path_exists(positive_path, path_cache, path_cache_lock):
        counts["path_stats"]["positive_paths_missing"] += 1
        counts["path_stats"]["all_paths_missing"] += 1
        counts["skip_reasons"]["positive_path_missing"] += 1
        review_rows.append(
            {
                "example_index": example_index,
                "query": query,
                "positive_path": positive_path,
                "candidate_rank": None,
                "candidate_path": positive_path,
                "candidate_score": None,
                "answer": None,
                "verdict": "MISSING_FILE",
                "path_role": "positive",
                "skip_reason": "positive_path_missing",
            }
        )
        return {
            "kept_row": None,
            "review_rows": review_rows,
            "counts": counts,
            "skipped_reason": "positive_path_missing",
        }

    try:
        positive_answer = answer_question(
            client_ctx, args.model, query, positive_path, args.max_retries
        )
    except ApiRequestError as exc:
        counts["skip_reasons"]["api_error"] += 1
        review_rows.append(
            {
                "example_index": example_index,
                "query": query,
                "positive_path": positive_path,
                "candidate_rank": None,
                "candidate_path": positive_path,
                "candidate_score": None,
                "answer": None,
                "verdict": "API_ERROR",
                "path_role": "positive",
                "skip_reason": "api_error",
                "error": str(exc),
            }
        )
        return {
            "kept_row": None,
            "review_rows": review_rows,
            "counts": counts,
            "skipped_reason": "api_error",
        }
    time.sleep(args.sleep_seconds)

    normalized_positive_answer = normalize_answer(positive_answer)
    if normalized_positive_answer == "CANNOT_ANSWER":
        positive_verdict = "CANNOT_ANSWER"
    else:
        try:
            positive_verdict = judge_answer(
                client_ctx,
                args.model,
                query,
                positive_path,
                positive_answer,
                args.max_retries,
            )
        except ApiRequestError as exc:
            counts["skip_reasons"]["api_error"] += 1
            review_rows.append(
                {
                    "example_index": example_index,
                    "query": query,
                    "positive_path": positive_path,
                    "candidate_rank": None,
                    "candidate_path": positive_path,
                    "candidate_score": None,
                    "answer": positive_answer,
                    "verdict": "API_ERROR",
                    "path_role": "positive",
                    "skip_reason": "api_error",
                    "error": str(exc),
                }
            )
            return {
                "kept_row": None,
                "review_rows": review_rows,
                "counts": counts,
                "skipped_reason": "api_error",
            }
        time.sleep(args.sleep_seconds)

    review_rows.append(
        {
            "example_index": example_index,
            "query": query,
            "positive_path": positive_path,
            "candidate_rank": None,
            "candidate_path": positive_path,
            "candidate_score": None,
            "answer": positive_answer,
            "verdict": positive_verdict,
            "path_role": "positive",
        }
    )
    if positive_verdict != "CORRECT":
        counts["skip_reasons"]["positive_not_correct"] += 1
        return {
            "kept_row": None,
            "review_rows": review_rows,
            "counts": counts,
            "skipped_reason": "positive_not_correct",
        }

    saw_missing_candidate = False
    for candidate in get_candidates(item, args.candidate_k):
        image_path = candidate["path"]
        counts["path_stats"]["candidate_paths_checked"] += 1
        counts["path_stats"]["all_paths_checked"] += 1
        if not path_exists(image_path, path_cache, path_cache_lock):
            counts["path_stats"]["candidate_paths_missing"] += 1
            counts["path_stats"]["all_paths_missing"] += 1
            saw_missing_candidate = True
            review_rows.append(
                {
                    "example_index": example_index,
                    "query": query,
                    "positive_path": item["chunk_path"],
                    "candidate_rank": candidate["rank"],
                    "candidate_path": image_path,
                    "candidate_score": candidate.get("score"),
                    "answer": None,
                    "verdict": "MISSING_FILE",
                    "path_role": "candidate",
                    "skip_reason": "candidate_path_missing",
                }
            )
            continue
        print(f"[{example_index:03d}] rank={candidate['rank']} answering", flush=True)
        try:
            answer = answer_question(
                client_ctx, args.model, query, image_path, args.max_retries
            )
        except MissingImageError:
            counts["path_stats"]["candidate_paths_missing"] += 1
            counts["path_stats"]["all_paths_missing"] += 1
            saw_missing_candidate = True
            review_rows.append(
                {
                    "example_index": example_index,
                    "query": query,
                    "positive_path": item["chunk_path"],
                    "candidate_rank": candidate["rank"],
                    "candidate_path": image_path,
                    "candidate_score": candidate.get("score"),
                    "answer": None,
                    "verdict": "MISSING_FILE",
                    "path_role": "candidate",
                    "skip_reason": "candidate_path_missing_race",
                }
            )
            continue
        except ApiRequestError as exc:
            counts["skip_reasons"]["api_error"] += 1
            review_rows.append(
                {
                    "example_index": example_index,
                    "query": query,
                    "positive_path": item["chunk_path"],
                    "candidate_rank": candidate["rank"],
                    "candidate_path": image_path,
                    "candidate_score": candidate.get("score"),
                    "answer": None,
                    "verdict": "API_ERROR",
                    "path_role": "candidate",
                    "skip_reason": "api_error",
                    "error": str(exc),
                }
            )
            return {
                "kept_row": None,
                "review_rows": review_rows,
                "counts": counts,
                "skipped_reason": "api_error",
            }
        time.sleep(args.sleep_seconds)

        normalized_answer = normalize_answer(answer)
        if normalized_answer == "CANNOT_ANSWER":
            verdict = "CANNOT_ANSWER"
        else:
            print(f"[{example_index:03d}] rank={candidate['rank']} judging", flush=True)
            try:
                verdict = judge_answer(
                    client_ctx, args.model, query, image_path, answer, args.max_retries
                )
            except MissingImageError:
                counts["path_stats"]["candidate_paths_missing"] += 1
                counts["path_stats"]["all_paths_missing"] += 1
                saw_missing_candidate = True
                review_rows.append(
                    {
                        "example_index": example_index,
                        "query": query,
                        "positive_path": item["chunk_path"],
                        "candidate_rank": candidate["rank"],
                        "candidate_path": image_path,
                        "candidate_score": candidate.get("score"),
                        "answer": answer,
                        "verdict": "MISSING_FILE",
                        "path_role": "candidate",
                        "skip_reason": "candidate_path_missing_race",
                    }
                )
                continue
            except ApiRequestError as exc:
                counts["skip_reasons"]["api_error"] += 1
                review_rows.append(
                    {
                        "example_index": example_index,
                        "query": query,
                        "positive_path": item["chunk_path"],
                        "candidate_rank": candidate["rank"],
                        "candidate_path": image_path,
                        "candidate_score": candidate.get("score"),
                        "answer": answer,
                        "verdict": "API_ERROR",
                        "path_role": "candidate",
                        "skip_reason": "api_error",
                        "error": str(exc),
                    }
                )
                return {
                    "kept_row": None,
                    "review_rows": review_rows,
                    "counts": counts,
                    "skipped_reason": "api_error",
                }
            time.sleep(args.sleep_seconds)

        counts["candidate_verdicts"][verdict] += 1
        review_rows.append(
            {
                "example_index": example_index,
                "query": query,
                "positive_path": item["chunk_path"],
                "candidate_rank": candidate["rank"],
                "candidate_path": image_path,
                "candidate_score": candidate.get("score"),
                "answer": answer,
                "verdict": verdict,
                "path_role": "candidate",
            }
        )

        if verdict != "CORRECT":
            selected_hns.append(image_path)
            if len(selected_hns) >= args.num_hard_negatives:
                break

    if saw_missing_candidate:
        counts["examples_with_missing_candidate_paths"] += 1

    kept_row = None
    skipped_reason = None
    if len(selected_hns) >= args.num_hard_negatives:
        kept_row = {
            "query": item["query"],
            "chunk_path": item["chunk_path"],
            "neg_chunk_paths": selected_hns,
            "source_positive_rank": item.get("positive_rank"),
            "source_positive_score": item.get("positive_score"),
        }
    else:
        counts["skip_reasons"]["not_enough_hard_negatives"] += 1
        skipped_reason = "not_enough_hard_negatives"
    return {
        "kept_row": kept_row,
        "review_rows": review_rows,
        "counts": counts,
        "skipped_reason": skipped_reason,
    }


def merge_counts(summary: dict, counts: dict) -> None:
    for verdict, count in counts["candidate_verdicts"].items():
        summary["candidate_verdicts"][verdict] += count
    for key, count in counts["path_stats"].items():
        summary["path_stats"][key] += count
    for key, count in counts["skip_reasons"].items():
        summary["skip_reasons"][key] += count
    summary["examples_with_missing_candidate_paths"] += counts[
        "examples_with_missing_candidate_paths"
    ]


def build_usage_summary(client_ctx: dict, model: str) -> dict:
    usage = dict(client_ctx["usage"])
    pricing = MODEL_PRICING.get(model)
    estimated_cost_usd = None
    if pricing:
        estimated_cost_usd = (
            usage["prompt_tokens"] / 1_000_000 * pricing["input_per_m"]
            + usage["completion_tokens"] / 1_000_000 * pricing["output_per_m"]
        )
    return {
        **usage,
        "provider": client_ctx["provider"],
        "model": model,
        "estimated_cost_usd": estimated_cost_usd,
    }


def main() -> int:
    args = parse_args()
    try:
        client_ctx = build_vlm_client(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    reviews_path = Path(args.reviews_output) if args.reviews_output else None
    summary_path = Path(args.summary_output) if args.summary_output else None

    summary = {
        "offset": args.offset,
        "limit": args.limit,
        "input_examples": 0,
        "completed_examples": 0,
        "kept_examples": 0,
        "skipped_examples": 0,
        "candidate_verdicts": {"CORRECT": 0, "WRONG": 0, "CANNOT_ANSWER": 0},
        "path_stats": {
            "positive_paths_checked": 0,
            "positive_paths_missing": 0,
            "candidate_paths_checked": 0,
            "candidate_paths_missing": 0,
            "all_paths_checked": 0,
            "all_paths_missing": 0,
            "missing_path_ratio": 0.0,
            "positive_missing_ratio": 0.0,
            "candidate_missing_ratio": 0.0,
        },
        "skip_reasons": {
            "not_enough_hard_negatives": 0,
            "positive_path_missing": 0,
            "positive_not_correct": 0,
            "api_error": 0,
        },
        "examples_with_missing_candidate_paths": 0,
        "token_usage": {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    path_cache: dict[str, bool] = {}
    path_cache_lock = threading.Lock()

    if reviews_path:
        reviews_path.parent.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, args.concurrency)
    max_pending = max_workers * 2

    with (
        output_path.open("w") as output_handle,
        (
            reviews_path.open("w") if reviews_path else open(os.devnull, "w")
        ) as reviews_handle,
    ):

        def handle_result(result: dict) -> None:
            kept_row = result["kept_row"]
            if kept_row is not None:
                append_jsonl(output_handle, kept_row)
                summary["kept_examples"] += 1
            else:
                summary["skipped_examples"] += 1
            for review_row in result["review_rows"]:
                if reviews_path:
                    append_jsonl(reviews_handle, review_row)
            merge_counts(summary, result["counts"])
            summary["completed_examples"] += 1
            path_stats = summary["path_stats"]
            if path_stats["all_paths_checked"] > 0:
                path_stats["missing_path_ratio"] = (
                    path_stats["all_paths_missing"] / path_stats["all_paths_checked"]
                )
            if path_stats["positive_paths_checked"] > 0:
                path_stats["positive_missing_ratio"] = (
                    path_stats["positive_paths_missing"]
                    / path_stats["positive_paths_checked"]
                )
            if path_stats["candidate_paths_checked"] > 0:
                path_stats["candidate_missing_ratio"] = (
                    path_stats["candidate_paths_missing"]
                    / path_stats["candidate_paths_checked"]
                )
            if summary_path:
                summary["token_usage"] = build_usage_summary(client_ctx, args.model)
                write_summary(summary_path, summary)
            print(
                "[progress] "
                f"completed={summary['completed_examples']} "
                f"kept={summary['kept_examples']} "
                f"skipped={summary['skipped_examples']} "
                f"missing_paths={path_stats['all_paths_missing']}/{path_stats['all_paths_checked']}",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = set()
            for i, item in enumerate(
                iter_jsonl(input_path, args.offset, args.limit), start=1
            ):
                summary["input_examples"] += 1
                pending.add(
                    executor.submit(
                        process_example,
                        i,
                        item,
                        args,
                        client_ctx,
                        path_cache,
                        path_cache_lock,
                    )
                )
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        handle_result(future.result())

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    handle_result(future.result())

    if summary_path:
        summary["token_usage"] = build_usage_summary(client_ctx, args.model)
        write_summary(summary_path, summary)
    else:
        summary["token_usage"] = build_usage_summary(client_ctx, args.model)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote: {output_path}")
    if reviews_path:
        print(f"Wrote: {reviews_path}")
    if summary_path:
        print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
