#!/usr/bin/env python3
"""Filter false-negative text retrieval candidates into usable hard negatives.

Input format: JSONL rows containing at least:
- query
- article_id
- chunk_index
- passage
- retrieve_top20

For each example:
1. Ask the LLM to answer the query from the positive passage.
2. Judge that answer on the same passage.
3. Skip the example unless the positive passage is CORRECT.
4. Scan retrieve_top20 in rank order, skipping the positive chunk.
5. For up to K candidates, ask the LLM to answer from the candidate passage.
6. Judge the answer on the same candidate passage.
7. If judge == CORRECT -> false negative (FN), skip it.
8. If judge == WRONG or CANNOT_ANSWER -> hard negative (HN), keep it.

If fewer than N HNs are found within the first K non-positive candidates, the
example is skipped.
"""

from __future__ import annotations

import argparse
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
    "gpt-4.1-mini": {"input_per_m": 0.40, "output_per_m": 1.60},
    "gemini-2.0-flash-001": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-2.0-flash": {"input_per_m": 0.10, "output_per_m": 0.40},
    "gemini-2.5-flash-preview-04-17": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gemini-2.5-pro": {"input_per_m": 1.25, "output_per_m": 10.00},
    "gemini-2.5-pro-preview-03-25": {"input_per_m": 1.25, "output_per_m": 10.00},
    "gemini-3.1-pro-preview": {"input_per_m": 1.25, "output_per_m": 10.00},
}

ANSWER_PROMPT_TEMPLATE = """You are reading a text passage from Wikipedia.

Based ONLY on the passage below, answer the question.
If the answer is not stated in the passage, reply "CANNOT_ANSWER".

Question: {question}

Passage:
\"\"\"
{passage}
\"\"\"

Give a short, direct answer (just the answer, no explanation)."""

JUDGE_PROMPT_TEMPLATE = """You are validating a candidate answer against a Wikipedia passage.

Based ONLY on the passage below, classify the candidate answer to the question as exactly one of:
- CORRECT: the candidate answer is supported by the passage and is correct.
- WRONG: the passage contains enough information to tell that the candidate answer is wrong.
- CANNOT_ANSWER: the passage does not contain enough information to verify the candidate answer.

Question: {question}
Candidate answer: {candidate_answer}

Passage:
\"\"\"
{passage}
\"\"\"

Return exactly one token: CORRECT, WRONG, or CANNOT_ANSWER."""


class ApiRequestError(RuntimeError):
    """Raised when text LLM calls fail after retries."""


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
        default=7,
        help="Number of HNs to keep per example",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument(
        "--gemini", action="store_true", help="Alias for --provider gemini."
    )
    parser.add_argument("--gemini-project", default="wise-coyote-478119-h0")
    parser.add_argument("--gemini-location", default="global")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=16)
    return parser.parse_args()


def iter_jsonl(path: Path, offset: int, limit: int):
    yielded = 0
    with path.open(encoding="utf-8") as f:
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


def init_token_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "calls": 0,
    }


def update_usage(
    client_ctx: dict, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    with client_ctx["usage_lock"]:
        client_ctx["usage"]["prompt_tokens"] += int(prompt_tokens or 0)
        client_ctx["usage"]["completion_tokens"] += int(completion_tokens or 0)
        client_ctx["usage"]["calls"] += 1


def build_text_client(args: argparse.Namespace) -> dict:
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
    client_ctx: dict, model: str, prompt: str, max_retries: int
) -> str:
    headers = {
        "Authorization": f"Bearer {client_ctx['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
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
    client_ctx: dict, model: str, prompt: str, max_retries: int
) -> str:
    from google.genai.types import GenerateContentConfig

    config = GenerateContentConfig(temperature=0, max_output_tokens=128)
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
        except Exception as exc:
            if attempt == max_retries:
                raise ApiRequestError(f"{type(exc).__name__}: {exc}") from exc
            wait_seconds = min(2**attempt, 20)
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait_seconds = max(wait_seconds, 15)
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable")


def call_text_llm(client_ctx: dict, model: str, prompt: str, max_retries: int) -> str:
    if client_ctx["provider"] == "openai":
        return call_openai_chat_completions(client_ctx, model, prompt, max_retries)
    return call_gemini_generate_content(client_ctx, model, prompt, max_retries)


def answer_question(
    client_ctx: dict, model: str, question: str, passage: str, max_retries: int
) -> str:
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, passage=passage)
    return call_text_llm(client_ctx, model, prompt, max_retries)


def judge_answer(
    client_ctx: dict,
    model: str,
    question: str,
    passage: str,
    answer: str,
    max_retries: int,
) -> str:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, candidate_answer=answer, passage=passage
    )
    verdict = call_text_llm(client_ctx, model, prompt, max_retries)
    verdict = verdict.strip().upper().replace('"', "").replace("`", "")
    compact_verdict = verdict.replace("-", "_").replace(" ", "_")
    for candidate in ("CANNOT_ANSWER", "CORRECT", "WRONG"):
        if candidate in compact_verdict:
            return candidate
    if any(
        marker in verdict
        for marker in (
            "NOT ENOUGH INFORMATION",
            "CANNOT BE VERIFIED",
            "CANNOT VERIFY",
            "CANNOT DETERMINE",
            "UNABLE TO DETERMINE",
            "NOT STATED",
            "NOT PROVIDED",
            "NOT MENTIONED",
            "DOES NOT SAY",
            "DOESN'T SAY",
        )
    ):
        return "CANNOT_ANSWER"
    return "WRONG"


def normalize_answer(answer: str) -> str:
    return answer.strip().upper().replace('"', "").replace("`", "")


def init_counts() -> dict:
    return {
        "candidate_verdicts": {"CORRECT": 0, "WRONG": 0, "CANNOT_ANSWER": 0},
        "skip_reasons": {
            "not_enough_hard_negatives": 0,
            "positive_not_correct": 0,
            "api_error": 0,
        },
    }


def get_candidates(item: dict, candidate_k: int) -> list[dict]:
    positive_key = (item["article_id"], item["chunk_index"])
    out = []
    for cand in item.get("retrieve_top20", []):
        cand_key = (cand.get("article_id"), cand.get("chunk_index"))
        if cand_key == positive_key:
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
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


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


def process_example(
    example_index: int, item: dict, args: argparse.Namespace, client_ctx: dict
) -> dict:
    query = item["query"]
    selected_hns = []
    review_rows = []
    counts = init_counts()

    positive_passage = item["passage"]
    positive_answer = None
    try:
        positive_answer = answer_question(
            client_ctx, args.model, query, positive_passage, args.max_retries
        )
        time.sleep(args.sleep_seconds)
        if normalize_answer(positive_answer) == "CANNOT_ANSWER":
            positive_verdict = "CANNOT_ANSWER"
        else:
            positive_verdict = judge_answer(
                client_ctx,
                args.model,
                query,
                positive_passage,
                positive_answer,
                args.max_retries,
            )
            time.sleep(args.sleep_seconds)
    except ApiRequestError as exc:
        counts["skip_reasons"]["api_error"] += 1
        review_rows.append(
            {
                "example_index": example_index,
                "query": query,
                "candidate_rank": None,
                "candidate_article_id": item["article_id"],
                "candidate_chunk_index": item["chunk_index"],
                "candidate_score": item.get("positive_score"),
                "candidate_title": item.get("title_guess"),
                "candidate_url": None,
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
        }

    review_rows.append(
        {
            "example_index": example_index,
            "query": query,
            "candidate_rank": None,
            "candidate_article_id": item["article_id"],
            "candidate_chunk_index": item["chunk_index"],
            "candidate_score": item.get("positive_score"),
            "candidate_title": item.get("title_guess"),
            "candidate_url": None,
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
        }

    for candidate in get_candidates(item, args.candidate_k):
        candidate_text = candidate.get("text", "")
        answer = None
        try:
            answer = answer_question(
                client_ctx, args.model, query, candidate_text, args.max_retries
            )
            time.sleep(args.sleep_seconds)
            if normalize_answer(answer) == "CANNOT_ANSWER":
                verdict = "CANNOT_ANSWER"
            else:
                verdict = judge_answer(
                    client_ctx,
                    args.model,
                    query,
                    candidate_text,
                    answer,
                    args.max_retries,
                )
                time.sleep(args.sleep_seconds)
        except ApiRequestError as exc:
            counts["skip_reasons"]["api_error"] += 1
            review_rows.append(
                {
                    "example_index": example_index,
                    "query": query,
                    "candidate_rank": candidate.get("rank"),
                    "candidate_article_id": candidate.get("article_id"),
                    "candidate_chunk_index": candidate.get("chunk_index"),
                    "candidate_score": candidate.get("score"),
                    "candidate_title": candidate.get("title"),
                    "candidate_url": candidate.get("url"),
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
            }

        counts["candidate_verdicts"][verdict] += 1
        review_rows.append(
            {
                "example_index": example_index,
                "query": query,
                "candidate_rank": candidate.get("rank"),
                "candidate_article_id": candidate.get("article_id"),
                "candidate_chunk_index": candidate.get("chunk_index"),
                "candidate_score": candidate.get("score"),
                "candidate_title": candidate.get("title"),
                "candidate_url": candidate.get("url"),
                "answer": answer,
                "verdict": verdict,
                "path_role": "candidate",
            }
        )

        if verdict != "CORRECT":
            selected_hns.append(candidate)
            if len(selected_hns) >= args.num_hard_negatives:
                break

    kept_row = None
    if len(selected_hns) >= args.num_hard_negatives:
        kept_row = {
            **item,
            "neg_hits": selected_hns,
            "neg_passages": [cand.get("text", "") for cand in selected_hns],
            "source_positive_rank": item.get("positive_rank"),
            "source_positive_score": item.get("positive_score"),
        }
    else:
        counts["skip_reasons"]["not_enough_hard_negatives"] += 1

    return {
        "kept_row": kept_row,
        "review_rows": review_rows,
        "counts": counts,
    }


def merge_counts(summary: dict, counts: dict) -> None:
    for verdict, count in counts["candidate_verdicts"].items():
        summary["candidate_verdicts"][verdict] += count
    for key, count in counts["skip_reasons"].items():
        summary["skip_reasons"][key] += count


def main() -> int:
    args = parse_args()
    provider = "gemini" if args.gemini else args.provider
    if provider == "gemini" and args.model.startswith("gpt-"):
        print("Gemini provider requires a Gemini model name", file=sys.stderr)
        return 1
    if provider == "openai" and args.model.startswith("gemini-"):
        print("OpenAI provider requires a non-Gemini model name", file=sys.stderr)
        return 1
    try:
        client_ctx = build_text_client(args)
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
        "skip_reasons": {
            "not_enough_hard_negatives": 0,
            "positive_not_correct": 0,
            "api_error": 0,
        },
        "token_usage": {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reviews_path:
        reviews_path.parent.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, args.concurrency)
    max_pending = max_workers * 2

    with (
        output_path.open("w", encoding="utf-8") as output_handle,
        (
            reviews_path.open("w", encoding="utf-8")
            if reviews_path
            else open(os.devnull, "w")
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
            if summary_path:
                summary["token_usage"] = build_usage_summary(client_ctx, args.model)
                write_summary(summary_path, summary)
            print(
                "[progress] "
                f"completed={summary['completed_examples']} "
                f"kept={summary['kept_examples']} "
                f"skipped={summary['skipped_examples']}",
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

    summary["token_usage"] = build_usage_summary(client_ctx, args.model)
    if summary_path:
        write_summary(summary_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote: {output_path}")
    if reviews_path:
        print(f"Wrote: {reviews_path}")
    if summary_path:
        print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
