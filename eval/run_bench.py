"""Run QA benchmark evaluation with various retrieval strategies.

Supports: SimpleQA, NQ, NQ-Tables, EVQA, MMSearch, WorldVQA, SimpleVQA, etc.

Usage:
    # Naive (no retrieval)
    python run_bench.py --task simpleqa --model Qwen/Qwen3.5-4B-Instruct --no-think

    # Pixel retrieval via search API
    python run_bench.py --task simpleqa --model Qwen/Qwen3.5-4B-Instruct --local-api --no-think

    # Text retrieval via search API
    python run_bench.py --task simpleqa --model Qwen/Qwen3.5-4B-Instruct --text-api --no-think

    # OpenRouter API (no local vLLM)
    python run_bench.py --task simpleqa --model openai/gpt-4 --open-router --api-key sk-or-v1-xxx
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

from tqdm.asyncio import tqdm_asyncio

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("run_naive.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add agent root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib import (
    # Data
    load_simpleqa_wikipedia,
    extract_url_from_metadata,
    encode_screenshot,
    make_compressed_encoder,
    load_nq_data,
    load_triviaqa_data,
    load_nq_tables_data,
    load_piqa_data,
    load_hellaswag_data,
    load_commonsenseqa_data,
    load_openbookqa_data,
    load_arc_data,
)
from lib import LLMClient, build_messages, build_react_messages
from lib.model_config import get_model_config, get_output_filename
from lib.retrieval import (
    _get_query_image_path_for_example,
    _save_worldvqa_query_image,
    _save_task_query_image,
)
from lib.benchmarks import (
    load_encyclopedic_vqa_data,
    load_shortformqa_data,
    load_worldvqa_data,
    load_simplevqa_data,
    load_factualvqa_data,
    load_mmsearch_data,
    load_webqa_data,
    load_multimodalqa_data,
    SUPPORTED_TASKS as DATASET_REPOS,
)


_DEFAULT_SPLIT_FOR_TASK = {
    "simpleqa": "test",
    "simpleqa_verified": "verified",
    "encyclopedic_vqa": "test",
    "worldvqa": "test",
    "2wiki": "validation",
    "simplevqa": "test",
    "factualvqa": "test",
    "mmsearch": "end2end",
    "webqa": "test",
    "multimodalqa": "validation",
    "nq": "validation",
    "triviaqa": "validation",
    "nq_tables": "dev",
    "piqa": "validation",
    "hellaswag": "validation",
    "commonsense_qa": "validation",
    "openbookqa": "validation",
    "arc_easy": "validation",
    "arc_challenge": "validation",
}


def _fetch_status(api_url: str | None, timeout: float = 5.0) -> dict | None:
    """Fetch /status from a search-API URL for reproducibility tagging.

    Returns the JSON dict on success, or {"_error": str, "url": str} on failure
    (failure is recorded rather than raised so a missing service does not block the run).
    """
    if not api_url:
        return None
    import urllib.request

    base = api_url.rstrip("/")
    if base.endswith("/search"):
        base = base[: -len("/search")]
    status_url = base + "/status"
    try:
        with urllib.request.urlopen(status_url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001 — best-effort capture
        return {"_error": f"{type(e).__name__}: {e}", "url": status_url}


def _build_run_metadata(args, n_loaded: int) -> dict:
    """Build the per-run reproducibility tuple stamped into every JSONL record.

    See root CLAUDE.md "Reproducibility tagging" — every benchmark number must
    carry: dataset+split+n, reader, retriever+checkpoint, index path+vec+built_at,
    top-k, query instruction, grader.
    """
    import datetime
    import subprocess

    reader_top_k = (
        args.reader_top_k if args.reader_top_k is not None else args.retrieval_top_k
    )
    meta = {
        "schema_version": 1,
        "run_started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Dataset + split + n
        "task": args.task,
        "split": getattr(args, "nq_split", None)
        if args.task == "nq"
        else _DEFAULT_SPLIT_FOR_TASK.get(args.task, "unknown"),
        "num_examples_requested": args.num_examples,
        "num_examples_loaded": n_loaded,
        # Reader
        "reader_model": args.model,
        "reader_max_tokens": getattr(args, "max_tokens", None),
        "reader_no_think": getattr(args, "no_think", False),
        "reader_extra_instructions": getattr(args, "reader_extra_instructions", None),
        # Retrieval k vs reader k (decoupled)
        "retrieval_top_k": args.retrieval_top_k,
        "reader_top_k": reader_top_k,
        # Query instruction (verbatim)
        "query_instruction": getattr(args, "query_instruction", None),
        # Retrieval API URLs + their /status (captures index path, vec count, built_at, model)
        "local_api_url": getattr(args, "local_api_url", None),
        "text_api_url": getattr(args, "text_api_url", None),
        "local_api_status": _fetch_status(getattr(args, "local_api_url", None)),
        "text_api_status": _fetch_status(getattr(args, "text_api_url", None)),
        # Misc dataset flags that change semantics
        "verified": getattr(args, "verified", False),
        "no_wiki_filter": getattr(args, "no_wiki_filter", False),
    }
    try:
        meta["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        meta["git_commit"] = None
    return meta


async def process_example(
    llm_client: LLMClient,
    retriever,
    example: dict,
    semaphore: asyncio.Semaphore,
    output_file: str | None = None,
    progress_counter: dict | None = None,
    total_examples: int = 0,
    encode_image_fn=None,
    task_name: str = "simpleqa",
    tiles_dir: str | None = None,
    run_metadata: dict | None = None,
) -> dict | None:
    """Process a single example: retrieve -> build messages -> call LLM."""
    async with semaphore:
        try:
            example_id = example.get("id", "unknown")
            # logger.info(f"Starting processing example {example_id}")

            # 1. Retrieve (data preparation happens inside retriever if needed)
            logger.debug(f"Retrieving for example {example_id}")
            retrieval_start_time = time.time()
            retrieval_result = await retriever.retrieve(example["problem"], example)
            retrieval_time = time.time() - retrieval_start_time
            logger.debug(
                f"Retrieval complete for example {example_id} (took {retrieval_time:.2f}s)"
            )

            # 1a. Snapshot the full retrieved set BEFORE reader-side slicing.
            # The JSONL records the full set so downstream grading can re-derive k=1/2/3
            # from a single retrieval-top-k=K_max run without re-querying the index.
            retrieved_full_images = (
                list(retrieval_result.images) if retrieval_result.images else []
            )
            retrieved_full_image_urls = list(
                getattr(retrieval_result, "image_urls", []) or []
            )
            # 1b. Reader-side top-k (decoupled from retrieval-k). Slice in place so build_messages
            # and the LLM see only the first reader_top_k items.
            reader_top_k = (run_metadata or {}).get("reader_top_k")
            if (
                reader_top_k is not None
                and retrieval_result.images
                and reader_top_k < len(retrieval_result.images)
            ):
                retrieval_result.images = retrieval_result.images[:reader_top_k]
                if getattr(retrieval_result, "image_urls", None):
                    retrieval_result.image_urls = retrieval_result.image_urls[
                        :reader_top_k
                    ]
                urls = []
                seen_urls = set()
                for url in getattr(retrieval_result, "image_urls", []) or []:
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        urls.append(url)
                if not urls and retrieval_result.source_url:
                    for url in retrieval_result.source_url.split(", "):
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            urls.append(url)
                        if len(urls) >= reader_top_k:
                            break
                if urls:
                    retrieval_result.source_url = ", ".join(urls)

            # 1b. Attach query image so VLM sees it alongside retrieved tiles
            if not retrieval_result.query_image_path and retrieval_result.has_content:
                if task_name == "encyclopedic_vqa":
                    tiles_dir = getattr(retriever, "tiles_dir", None) or "tiles/evqa"
                    img_path = _get_query_image_path_for_example(
                        example, tiles_dir, quiet=True
                    )
                    if img_path:
                        retrieval_result.query_image_path = img_path
                elif task_name in (
                    "worldvqa",
                    "simplevqa",
                    "factualvqa",
                    "mmsearch",
                    "webqa",
                    "multimodalqa",
                ):
                    img_path = _save_task_query_image(
                        example, task_name, base_dir="tiles"
                    )
                    if img_path:
                        retrieval_result.query_image_path = img_path

            # 2. Build messages
            logger.debug(f"Building messages for example {example_id}")
            _encode_fn = (
                encode_image_fn if encode_image_fn is not None else encode_screenshot
            )
            messages = build_messages(
                query=example["problem"],
                retrieval_result=retrieval_result,
                encode_image_fn=_encode_fn,
                additional_instructions=example.get("additional_instructions"),
                few_shot_demos=example.get("_reader_few_shot"),
            )
            logger.debug(f"Messages built for example {example_id}")

            # 3. Call LLM
            # logger.info(f"Calling LLM for example {example_id}")
            llm_start_time = time.time()
            generated_text, usage = await llm_client.generate(messages)
            llm_time = time.time() - llm_start_time

            # Update progress counter
            if progress_counter is not None:
                progress_counter["completed"] += 1
                completed = progress_counter["completed"]
                ((completed / total_examples * 100) if total_examples > 0 else 0)

                # Accumulate timing stats
                if "retrieval_times" not in progress_counter:
                    progress_counter["retrieval_times"] = []
                    progress_counter["llm_times"] = []
                progress_counter["retrieval_times"].append(retrieval_time)
                progress_counter["llm_times"].append(llm_time)

            # 4. Build result
            result = {
                "example_id": example["id"],
                # 0-indexed position in the loaded examples list — see run_async() stamping.
                # Async writes append in completion order; sort downstream by load_index
                # to recover canonical load order and the strict line-level prefix property
                # (records with load_index < N are exactly the first N loaded examples).
                "load_index": example.get("_load_index"),
                "problem": example["problem"],
                "model": llm_client.model,
                "final_response": generated_text,
                "original_data": {
                    k: v
                    for k, v in example.items()
                    if not hasattr(v, "save") and not k.startswith("_")
                },
                "full_traces": {},
                "dataset_name": task_name,
                "retrieval_type": retrieval_result.retrieval_type,
                "has_retrieval_content": retrieval_result.has_content,
                "usage": usage,
                "success": True,
                "timing": {
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_time,
                    "total_time": retrieval_time + llm_time,
                },
                # Per-record reproducibility tag — see root CLAUDE.md "Reproducibility tagging".
                # Stamped on every record so any single line is self-describing.
                "run_metadata": run_metadata,
            }

            # Add retrieval-specific info
            if retrieval_result.source_url:
                result["used_url"] = retrieval_result.source_url
            if retrieval_result.text:
                result["context_length"] = len(retrieval_result.text)
            # `retrieved_images` records the FULL retrieved set (pre reader-side slicing)
            # so downstream grading at k=1/2/3 can be derived from one retrieval_top_k=K_max run.
            if retrieved_full_images:
                result["retrieved_images"] = []
                for idx, (path, score) in enumerate(retrieved_full_images):
                    item = {"path": path, "score": score}
                    if (
                        idx < len(retrieved_full_image_urls)
                        and retrieved_full_image_urls[idx]
                    ):
                        item["url"] = retrieved_full_image_urls[idx]
                    result["retrieved_images"].append(item)
            if retrieval_result.pixel_query_path:
                result["pixel_query_path"] = retrieval_result.pixel_query_path

            # Always include query image path in result for eval analysis
            query_img_path = (
                retrieval_result.query_image_path or retrieval_result.pixel_query_path
            )
            if not query_img_path:
                if task_name == "encyclopedic_vqa" and tiles_dir:
                    query_img_path = _get_query_image_path_for_example(
                        example, tiles_dir
                    )
                elif task_name == "worldvqa":
                    query_img_path = _save_worldvqa_query_image(
                        example, base_dir="tiles"
                    )
                elif task_name in (
                    "simplevqa",
                    "factualvqa",
                    "mmsearch",
                    "webqa",
                    "multimodalqa",
                ):
                    query_img_path = _save_task_query_image(
                        example, task_name, base_dir="tiles"
                    )
            if query_img_path:
                result["query_image_path"] = query_img_path

            # Record compressed image paths if pixel compression was used
            if (
                _encode_fn is not None
                and hasattr(_encode_fn, "compressed_paths")
                and retrieval_result.images
            ):
                compressed_images = []
                for orig_path, score in retrieval_result.images:
                    comp_path = _encode_fn.compressed_paths.get(orig_path)
                    if comp_path:
                        compressed_images.append(
                            {
                                "original_path": orig_path,
                                "compressed_path": comp_path,
                                "score": score,
                            }
                        )
                if compressed_images:
                    result["compressed_images"] = compressed_images
                    result["pixel_compress_ratio"] = _encode_fn.compress_ratio
                    result["compressed_images_dir"] = _encode_fn.save_dir

            # Incremental save
            if output_file:
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            return result

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            example_id = example.get("id", "unknown")

            # Update progress counter even on error
            if progress_counter is not None:
                progress_counter["completed"] += 1
                logger.warning(f"Example {example_id} failed: {e}")

            logger.error(f"Error processing {example_id}: {e}")
            logger.error(f"Traceback: {error_trace}")
            result = {
                "example_id": example.get("id"),
                "problem": example.get("problem"),
                "model": llm_client.model,
                "final_response": None,
                "original_data": {
                    k: v
                    for k, v in example.items()
                    if not hasattr(v, "save") and not k.startswith("_")
                },
                "dataset_name": task_name,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timing": {
                    "retrieval_time": None,
                    "llm_time": None,
                    "total_time": None,
                },
            }
            if output_file:
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
            return result


import re

_SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)


async def _local_api_search(
    api_url: str, query_text: str, top_k: int, nprobe: int | None = None
) -> list[dict]:
    """Single-query search against local API, returns hits."""
    import aiohttp

    payload = {"queries": [{"text": query_text}], "n_docs": top_k}
    if nprobe is not None:
        payload["nprobe"] = nprobe
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    return []
                result = await response.json()
                results_list = result.get("results", [])
                return results_list[0].get("hits", []) if results_list else []
    except Exception as e:
        logger.error(f"ReAct search failed: {e}")
        return []


def _hits_to_retrieval_result(hits: list[dict]) -> "RetrievalResult":  # noqa: F821
    """Convert API hits to RetrievalResult (same logic as LocalAPIRetriever)."""
    from lib.retrieval import RetrievalResult

    if not hits:
        return RetrievalResult(retrieval_type="local_api_react")
    images = []
    image_urls = []
    urls = []
    seen_urls = set()
    for hit in hits:
        path = hit.get("path", "")
        score = hit.get("score", 0.0)
        url = hit.get("url", "")
        if path and os.path.exists(path):
            images.append((path, score))
            image_urls.append(url or None)
        if url and url not in seen_urls:
            seen_urls.add(url)
            urls.append(url)
    return RetrievalResult(
        images=images,
        image_urls=image_urls,
        source_url=", ".join(urls) if urls else None,
        retrieval_type="local_api_react",
    )


async def process_example_react(
    llm_client: LLMClient,
    retriever,
    example: dict,
    semaphore: asyncio.Semaphore,
    output_file: str | None = None,
    progress_counter: dict | None = None,
    total_examples: int = 0,
    encode_image_fn=None,
    task_name: str = "simpleqa",
    tiles_dir: str | None = None,
    max_turns: int = 3,
    api_url: str = "http://localhost:30888/search",
    react_top_k: int = 5,
    nprobe: int | None = None,
    prompt_version: str = "v1",
) -> dict | None:
    """Process a single example with ReAct multi-turn retrieval.

    Flow: retrieve → LLM → if <search>query</search> in response → retrieve again → LLM → ...
    Stops when: (1) no <search> tag in response, (2) max_turns reached, or (3) error.
    """
    async with semaphore:
        try:
            example_id = example.get("id", "unknown")
            total_start = time.time()

            # Round 1: use the normal retriever (which may have prefetched results)
            retrieval_start = time.time()
            retrieval_result = await retriever.retrieve(example["problem"], example)
            retrieval_time = time.time() - retrieval_start

            retrieval_results = [retrieval_result]
            assistant_responses = []
            all_search_queries = []
            total_retrieval_time = retrieval_time
            total_llm_time = 0.0
            turns_used = 1

            _encode_fn = (
                encode_image_fn if encode_image_fn is not None else encode_screenshot
            )

            for turn in range(max_turns):
                is_last = turn == max_turns - 1
                # Build messages (multi-turn)
                messages = build_react_messages(
                    query=example["problem"],
                    retrieval_results=retrieval_results,
                    assistant_responses=assistant_responses,
                    encode_image_fn=_encode_fn,
                    prompt_version=prompt_version,
                    is_last_turn=is_last,
                    previous_queries=all_search_queries,
                )

                # Call LLM
                llm_start = time.time()
                generated_text, usage = await llm_client.generate(messages)
                total_llm_time += time.time() - llm_start

                # Check for <search> tag
                match = _SEARCH_TAG_RE.search(generated_text)
                if not match or is_last:
                    # Final answer (or last turn forced)
                    # Strip any remaining <search> tags from forced-stop responses
                    final_response = _SEARCH_TAG_RE.sub("", generated_text).strip()
                    turns_used = turn + 1
                    break

                # Extract search query and do another round
                search_query = match.group(1).strip()
                all_search_queries.append(search_query)
                assistant_responses.append(generated_text)
                logger.info(
                    f"ReAct [{example_id}] turn {turn + 1}: searching '{search_query[:80]}'"
                )

                # New retrieval
                ret_start = time.time()
                new_hits = await _local_api_search(
                    api_url, search_query, react_top_k, nprobe
                )
                total_retrieval_time += time.time() - ret_start
                retrieval_results.append(_hits_to_retrieval_result(new_hits))
            else:
                final_response = generated_text
                turns_used = max_turns

            # Update progress counter
            if progress_counter is not None:
                progress_counter["completed"] += 1
                if "retrieval_times" not in progress_counter:
                    progress_counter["retrieval_times"] = []
                    progress_counter["llm_times"] = []
                progress_counter["retrieval_times"].append(total_retrieval_time)
                progress_counter["llm_times"].append(total_llm_time)

            total_time = time.time() - total_start

            # Build per-turn traces (images + assistant response for each round)
            react_trace = []
            for turn_idx, rr in enumerate(retrieval_results):
                turn_info = {
                    "turn": turn_idx + 1,
                    "images": [
                        {"path": path, "score": score, "url": rr.source_url}
                        for path, score in rr.images
                    ],
                }
                if turn_idx < len(assistant_responses):
                    turn_info["assistant_response"] = assistant_responses[turn_idx]
                elif turn_idx == len(retrieval_results) - 1:
                    # Last turn: the final_response is the answer
                    turn_info["assistant_response"] = final_response
                react_trace.append(turn_info)

            # Build result
            result = {
                "example_id": example["id"],
                "problem": example["problem"],
                "model": llm_client.model,
                "final_response": final_response,
                "original_data": {
                    k: v
                    for k, v in example.items()
                    if not hasattr(v, "save") and not k.startswith("_")
                },
                "full_traces": {},
                "dataset_name": task_name,
                "retrieval_type": "local_api_react",
                "has_retrieval_content": any(r.has_content for r in retrieval_results),
                "usage": usage,
                "success": True,
                "react_turns": turns_used,
                "react_search_queries": all_search_queries,
                "react_trace": react_trace,
                "timing": {
                    "retrieval_time": total_retrieval_time,
                    "llm_time": total_llm_time,
                    "total_time": total_time,
                },
            }

            # Add retrieval info from first round
            if retrieval_results[0].source_url:
                result["used_url"] = retrieval_results[0].source_url
            if retrieval_results[0].images:
                result["retrieved_images"] = [
                    {"path": path, "score": score}
                    for path, score in retrieval_results[0].images
                ]
            # All retrieved images across all rounds
            all_images = []
            for rr in retrieval_results:
                for path, score in rr.images:
                    all_images.append({"path": path, "score": score})
            if len(retrieval_results) > 1:
                result["all_retrieved_images"] = all_images

            # Incremental save
            if output_file:
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            return result

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            example_id = example.get("id", "unknown")

            if progress_counter is not None:
                progress_counter["completed"] += 1
                logger.warning(f"ReAct example {example_id} failed: {e}")

            logger.error(f"Error processing (react) {example_id}: {e}")
            logger.error(f"Traceback: {error_trace}")
            result = {
                "example_id": example.get("id"),
                "problem": example.get("problem"),
                "model": llm_client.model,
                "final_response": None,
                "original_data": {
                    k: v
                    for k, v in example.items()
                    if not hasattr(v, "save") and not k.startswith("_")
                },
                "dataset_name": task_name,
                "retrieval_type": "local_api_react",
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timing": {
                    "retrieval_time": None,
                    "llm_time": None,
                    "total_time": None,
                },
            }
            if output_file:
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
            return result


def print_statistics(results: list[dict], args) -> None:
    """Print evaluation statistics."""
    total = len(results)
    if total == 0:
        print("No results to report.")
        return

    # Count success/failure
    success_count = sum(1 for r in results if r.get("success", False))
    failure_count = total - success_count

    print("-" * 40)
    print(f"Total: {total} examples")
    print(f"  Success: {success_count} ({success_count / total * 100:.1f}%)")
    print(f"  Failed: {failure_count} ({failure_count / total * 100:.1f}%)")

    # Timing statistics
    successful_results = [
        r for r in results if r.get("success", False) and r.get("timing")
    ]
    if successful_results:
        retrieval_times = [
            r["timing"]["retrieval_time"]
            for r in successful_results
            if r["timing"].get("retrieval_time") is not None
        ]
        llm_times = [
            r["timing"]["llm_time"]
            for r in successful_results
            if r["timing"].get("llm_time") is not None
        ]
        total_times = [
            r["timing"]["total_time"]
            for r in successful_results
            if r["timing"].get("total_time") is not None
        ]

        if retrieval_times:
            print(
                f"\nTiming Statistics (for {len(successful_results)} successful requests):"
            )
            print("  Jina read time:")
            print(f"    Mean: {sum(retrieval_times) / len(retrieval_times):.2f}s")
            print(f"    Min: {min(retrieval_times):.2f}s")
            print(f"    Max: {max(retrieval_times):.2f}s")
            print(
                f"    Median: {sorted(retrieval_times)[len(retrieval_times) // 2]:.2f}s"
            )

        if llm_times:
            print("  LLM call time:")
            print(f"    Mean: {sum(llm_times) / len(llm_times):.2f}s")
            print(f"    Min: {min(llm_times):.2f}s")
            print(f"    Max: {max(llm_times):.2f}s")
            print(f"    Median: {sorted(llm_times)[len(llm_times) // 2]:.2f}s")

        if total_times:
            print("  Total time:")
            print(f"    Mean: {sum(total_times) / len(total_times):.2f}s")
            print(f"    Min: {min(total_times):.2f}s")
            print(f"    Max: {max(total_times):.2f}s")
            print(f"    Median: {sorted(total_times)[len(total_times) // 2]:.2f}s")

    # Count by retrieval type (only for successful)
    successful = [r for r in results if r.get("success", False)]
    if successful:
        type_counts = {}
        for r in successful:
            rt = r.get("retrieval_type", "unknown")
            type_counts[rt] = type_counts.get(rt, 0) + 1

        print("\nRetrieval types (successful only):")
        for rt, count in type_counts.items():
            print(f"  {rt}: {count} ({count / len(successful) * 100:.1f}%)")

    # Retrieval accuracy (for vector mode)
    # Checks if any of the top-k retrieved tiles come from the correct Wikipedia page
    if args.retrieval_augment or args.use_tiled_retrieval or args.local_api:
        retrieval_results = [r for r in results if r.get("retrieved_images")]
        if retrieval_results:
            correct = 0
            for r in retrieval_results:
                # Try to get ground truth URL from metadata
                gt_url = extract_url_from_metadata(r.get("original_data", {}))

                if not gt_url:
                    # Fallback: match by example_id in tile filename
                    example_id = r.get("example_id", "")
                    for img_info in r.get("retrieved_images", []):
                        img_path = img_info.get("original_path") or img_info.get(
                            "path", ""
                        )
                        img_basename = os.path.basename(img_path)
                        if example_id in img_basename:
                            correct += 1
                            break
                else:
                    # Check if any retrieved tile's URL matches the ground truth URL
                    # retrieved_url is a string of comma-separated URLs from tiles
                    retrieved_url = r.get("used_url", "")
                    # Check if the ground truth URL is contained in the retrieved URLs
                    if gt_url in retrieved_url:
                        correct += 1

            print("\nRetrieval Accuracy:")
            print(
                f"  Correct (top-{args.retrieval_top_k}): {correct}/{len(retrieval_results)} ({correct / len(retrieval_results) * 100:.1f}%)"
            )

    # ReAct turn statistics
    react_results = [r for r in results if r.get("react_turns") is not None]
    if react_results:
        turns = [r["react_turns"] for r in react_results]
        from collections import Counter

        turn_counts = Counter(turns)
        print("\nReAct Turn Distribution:")
        for t in sorted(turn_counts):
            print(
                f"  {t} turn(s): {turn_counts[t]} ({turn_counts[t] / len(react_results) * 100:.1f}%)"
            )
        print(f"  Average turns: {sum(turns) / len(turns):.2f}")
        multi_turn = sum(1 for t in turns if t > 1)
        print(
            f"  Examples needing re-search: {multi_turn}/{len(react_results)} ({multi_turn / len(react_results) * 100:.1f}%)"
        )

    print("-" * 40)
    print(f"Results saved to {args.output}")


async def run_async(args):
    """Main async entry point."""
    # 1. Load data
    if args.task == "simpleqa":
        examples = load_simpleqa_wikipedia(
            args.num_examples,
            verified=args.verified,
            no_wiki_filter=getattr(args, "no_wiki_filter", False),
        )
    elif args.task == "encyclopedic_vqa":
        split = args.subset or "val"
        examples = load_encyclopedic_vqa_data(
            split,
            args.num_examples,
            dataset_filter=args.evqa_dataset_filter,
            question_type_filter=args.evqa_question_type_filter,
            local_path=args.evqa_data_path,
        )
        if args.evqa_instruction_override is not None:
            for ex in examples:
                ex["additional_instructions"] = args.evqa_instruction_override
    elif args.task == "worldvqa":
        examples = load_worldvqa_data(
            args.num_examples, language_filter=getattr(args, "worldvqa_language", None)
        )
    elif args.task == "2wiki":
        dataset_repo = DATASET_REPOS["2wiki"]
        examples = load_shortformqa_data(dataset_repo, args.num_examples)
    elif args.task == "simplevqa":
        examples = load_simplevqa_data(args.num_examples)
    elif args.task == "factualvqa":
        examples = load_factualvqa_data(args.num_examples)
    elif args.task == "mmsearch":
        examples = load_mmsearch_data(args.num_examples)
    elif args.task == "webqa":
        examples = load_webqa_data(args.num_examples)
    elif args.task == "multimodalqa":
        examples = load_multimodalqa_data(args.num_examples)
    elif args.task == "nq":
        examples = load_nq_data(
            args.num_examples, split=getattr(args, "nq_split", "validation")
        )
    elif args.task == "triviaqa":
        examples = load_triviaqa_data(args.num_examples)
    elif args.task == "nq_tables":
        examples = load_nq_tables_data(args.num_examples)
    elif args.task == "piqa":
        examples = load_piqa_data(args.num_examples)
    elif args.task == "hellaswag":
        examples = load_hellaswag_data(args.num_examples)
    elif args.task == "commonsense_qa":
        examples = load_commonsenseqa_data(args.num_examples)
    elif args.task == "openbookqa":
        examples = load_openbookqa_data(args.num_examples)
    elif args.task == "arc_easy":
        examples = load_arc_data("ARC-Easy", args.num_examples)
    elif args.task == "arc_challenge":
        examples = load_arc_data("ARC-Challenge", args.num_examples)
    else:
        raise ValueError(f"Unsupported task: {args.task}.")

    # Stamp each example with its 0-indexed position in the loaded list so
    # process_example() can record it. Async writes append in completion order, not
    # load order — load_index lets downstream `sorted(records, key=lambda r: r["load_index"])`
    # recover the canonical order and gives a true line-level prefix property
    # (n=200 records are exactly load_index ∈ [0, 200) of an n=1000 run).
    for _idx, _ex in enumerate(examples):
        _ex["_load_index"] = _idx

    # Build the per-run reproducibility metadata once, after dataset is loaded so
    # we know n_loaded. Stamped on every JSONL record by process_example().
    run_metadata = _build_run_metadata(args, n_loaded=len(examples))
    print(
        f"\n[run_metadata] task={run_metadata['task']} split={run_metadata['split']} "
        f"n_requested={run_metadata['num_examples_requested']} n_loaded={run_metadata['num_examples_loaded']} "
        f"retrieval_top_k={run_metadata['retrieval_top_k']} reader_top_k={run_metadata['reader_top_k']} "
        f"reader={run_metadata['reader_model']}"
    )
    for api_key in ("local_api_status", "text_api_status"):
        st = run_metadata.get(api_key)
        if st and "_error" not in st:
            print(
                f"[run_metadata] {api_key}: vec={st.get('total_vectors')} "
                f"built_at={st.get('index_built_at')} model={st.get('model')}"
            )
        elif st:
            print(f"[run_metadata] {api_key}: ERROR {st.get('_error')}")

    if args.task in ("nq", "triviaqa", "nq_tables"):
        for ex in examples:
            ex["additional_instructions"] = (
                "Answer with as few words as possible. Give only the answer, no explanation."
            )

    if args.reader_extra_instructions:
        for ex in examples:
            base = ex.get("additional_instructions") or ""
            ex["additional_instructions"] = (
                base + "\n\n" + args.reader_extra_instructions
            ).strip()

    if args.reader_few_shot_json:
        with open(args.reader_few_shot_json) as _fsf:
            _demos = json.load(_fsf)
        for ex in examples:
            ex["_reader_few_shot"] = _demos
        logger.info(
            f"Loaded {len(_demos)} few-shot demo(s) from {args.reader_few_shot_json}"
        )

    # Get model configuration
    model_config = get_model_config(args.model)

    # Handle OpenRouter API
    if args.open_router:
        api_base = "https://openrouter.ai/api/v1"
        if args.api_key and args.api_key != "dummy":
            api_key = args.api_key
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "dummy":
            raise ValueError(
                "OpenRouter API key required. Set --api-key or OPENROUTER_API_KEY environment variable."
            )
        logger.info(f"Using OpenRouter API with model: {args.model}")
        model = args.model
    elif args.commonstack:
        api_base = "https://api.commonstack.ai/v1"
        if args.api_key and args.api_key != "dummy":
            api_key = args.api_key
        else:
            api_key = os.getenv("COMMONSTACK_API_KEY")
        if not api_key or api_key == "dummy":
            raise ValueError(
                "Commonstack API key required. Set --api-key or COMMONSTACK_API_KEY environment variable."
            )
        logger.info(f"Using Commonstack API with model: {args.model}")
        model = args.model
    else:
        # Override with command-line args if provided
        # For Gemini, api_base from config is None, so use command-line arg or default
        api_base = (
            args.api_base
            if args.api_base
            else (model_config["api_base"] or "http://localhost:8000/v1")
        )
        api_key = args.api_key if args.api_key else model_config["api_key"]
        model = model_config["model"]

    # Generate output filename with model name if output is not explicitly set
    if not args.output or args.output == "auto":
        # Determine mode for filename
        if args.url_screenshot:
            mode_str = "screenshot"
        elif args.url_tiled_screenshot:
            mode_str = "tiled_screenshot"
        elif args.url_text:
            mode_str = f"text_{args.text_source}"
        elif args.retrieval_augment:
            if args.use_colqwen_retrieval:
                mode_str = "vector_colqwen"
            else:
                mode_str = "vector_jina"
        elif args.use_tiled_retrieval:
            if args.use_colqwen_retrieval:
                mode_str = "tiled_vector_colqwen"
            elif args.use_qwen3vl_embedding:
                mode_str = "tiled_vector_qwen3vl_embedding"
                if args.task == "encyclopedic_vqa":
                    if args.evqa_multimodal_query:
                        if args.evqa_multimodal_query_text_only:
                            mode_str += "_multimodal_textonly"
                        elif args.evqa_multimodal_query_image_only:
                            mode_str += "_multimodal_imageonly"
                        else:
                            mode_str += "_multimodal"
                    else:
                        mode_str += "_querycard"
                elif args.pixel_query:
                    mode_str += "_pixelq"
                if args.pixel_compress_ratio and args.pixel_compress_ratio > 1:
                    mode_str += f"_compress{args.pixel_compress_ratio}x"
            else:
                mode_str = "tiled_vector_jina"
        elif args.text_api:
            mode_str = "text_api"
        elif args.html_dom_lookup:
            mode_str = "html_dom_lookup"
        elif args.hybrid:
            mode_str = "hybrid"
        elif args.text_vector:
            if args.text_source == "ds-serve":
                mode_str = "text_vector_ds_serve"
            else:
                mode_str = f"text_vector_{args.text_source}_{args.text_embed_preset}"
        else:
            mode_str = (
                "no_retrieval"
                if args.task in ("encyclopedic_vqa", "worldvqa")
                else "naive"
            )
            if args.task == "2wiki":
                mode_str = "naive"

        output_dir = "eval_output"
        args.output = get_output_filename(
            output_dir=output_dir,
            model_name=model,
            mode=mode_str,
            num_examples=args.num_examples or len(examples),
            url_screenshot=args.url_screenshot,
            task=args.task,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check if output file exists
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        if not args.force:
            print(
                f"Error: Output file '{args.output}' already exists and is not empty."
            )
            print("Use --force to overwrite.")
            sys.exit(1)
        else:
            print(f"Warning: Overwriting existing file '{args.output}'")

    # Clear output file
    with open(args.output, "w"):
        pass

    # 2. Initialize retriever (each retriever uses data layer internally)
    # Tile width is fixed to 1024 (matches screenshot width)
    TILE_WIDTH = 1024

    # Set default tiles_dir if not specified
    if args.tiles_dir is None:
        args.tiles_dir = f"tiles-{TILE_WIDTH}x{args.tile_height}"

    # Calculate max_tiles from context length if not specified
    # Qwen3-VL: 1024x1024 tile = 1024 image tokens + ~10 overhead = ~1034 tokens
    # Scale by tile height ratio
    BASE_TOKENS_PER_TILE = 1050  # for 1024x1024
    TOKENS_PER_TILE = int(BASE_TOKENS_PER_TILE * args.tile_height / 1024)
    RESERVED_TOKENS = 2000  # For question and response
    if args.max_tiles is None and (
        args.url_tiled_screenshot or args.use_tiled_retrieval
    ):
        available_tokens = args.model_context_length - RESERVED_TOKENS
        args.max_tiles = max(1, available_tokens // TOKENS_PER_TILE)
        logger.info(
            f"Auto-calculated max_tiles: {args.max_tiles} (context={args.model_context_length}, per_tile={TOKENS_PER_TILE}, tile={TILE_WIDTH}x{args.tile_height})"
        )

    from lib.retrievers import build_retriever

    retriever, mode = build_retriever(args, examples, model, api_base, api_key)
    # (retriever selection logic moved to simpleqa/retriever_factory.py)
    # 3. Initialize LLM client
    llm_client = LLMClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        max_context_tokens=args.model_context_length,
        timeout=args.timeout,
        enable_thinking=(False if args.no_think else None),
        force_openai_compat=(args.open_router or args.commonstack),
    )

    # 3b. Create pixel-compressed encoder for generation if requested
    gen_encode_fn = None
    if args.pixel_compress_ratio and args.pixel_compress_ratio > 1:
        gen_encode_fn = make_compressed_encoder(args.pixel_compress_ratio)
        mode += f" (PixelCompress={args.pixel_compress_ratio}x)"
        logger.info(f"Generation pixel compression: {args.pixel_compress_ratio}x")

    # 3c. Prefetch retrieval results for batch-capable retrievers
    if hasattr(retriever, "prefetch"):
        print("Prefetching retrieval results (batch API call)...")
        await retriever.prefetch(examples)

    # 4. Process examples
    total_examples = len(examples)
    logger.info(
        f"Processing {total_examples} examples (Mode: {mode}, Concurrency: {args.max_concurrent})"
    )
    print(f"\n{'=' * 80}")
    print(
        f"Starting evaluation: {total_examples} examples with max {args.max_concurrent} concurrent requests"
    )
    if gen_encode_fn:
        print(
            f"Pixel compression for generation: {args.pixel_compress_ratio}x (retrieval at original resolution)"
        )
    print(f"{'=' * 80}\n")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Progress counter (shared dict for async updates)
    progress_counter = {"completed": 0, "start_time": time.time()}

    tiles_dir = getattr(retriever, "tiles_dir", None) or (
        args.tiles_dir if hasattr(args, "tiles_dir") else None
    )
    if args.react and args.local_api:
        tasks = [
            process_example_react(
                llm_client,
                retriever,
                ex,
                semaphore,
                args.output,
                progress_counter,
                total_examples,
                encode_image_fn=gen_encode_fn,
                task_name=args.task,
                tiles_dir=tiles_dir,
                max_turns=args.react_max_turns,
                api_url=args.local_api_url,
                react_top_k=args.retrieval_top_k,
                nprobe=args.nprobe,
                prompt_version=args.react_prompt,
            )
            for ex in examples
        ]
    else:
        tasks = [
            process_example(
                llm_client,
                retriever,
                ex,
                semaphore,
                args.output,
                progress_counter,
                total_examples,
                encode_image_fn=gen_encode_fn,
                task_name=args.task,
                tiles_dir=tiles_dir,
                run_metadata=run_metadata,
            )
            for ex in examples
        ]

    results = await tqdm_asyncio.gather(*tasks)

    # Print completion summary
    elapsed_time = time.time() - progress_counter["start_time"]
    print(f"\n{'=' * 80}")
    print(
        f"Evaluation completed: {progress_counter['completed']}/{total_examples} examples in {elapsed_time:.1f}s"
    )
    print(
        f"Average time per example: {elapsed_time / max(1, progress_counter['completed']):.2f}s"
    )
    print(f"{'=' * 80}\n")

    # 5. Print statistics
    print_statistics(results, args)


def main():
    parser = argparse.ArgumentParser(
        description="Run SimpleQA evaluation with various retrieval strategies"
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="simpleqa",
        choices=[
            "simpleqa",
            "encyclopedic_vqa",
            "worldvqa",
            "2wiki",
            "simplevqa",
            "factualvqa",
            "mmsearch",
            "webqa",
            "multimodalqa",
            "nq",
            "triviaqa",
            "nq_tables",
            "piqa",
            "hellaswag",
            "commonsense_qa",
            "openbookqa",
            "arc_easy",
            "arc_challenge",
        ],
        help="Task/benchmark to run (default: simpleqa)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset (e.g., 'val' or 'test' for encyclopedic_vqa)",
    )
    parser.add_argument(
        "--nq-split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="NQ only: HuggingFace split to stream (default: validation)",
    )
    parser.add_argument(
        "--evqa-dataset-filter",
        type=str,
        default=None,
        choices=["inaturalist", "landmarks"],
        help="EVQA only: filter by dataset_name ('inaturalist' or 'landmarks')",
    )
    parser.add_argument(
        "--evqa-question-type-filter",
        type=str,
        default=None,
        help="EVQA only: filter by question_type. Comma-separate to allow multiple "
        "(e.g. 'automatic,templated'). Valid values: templated, automatic, multi_answer, 2_hop.",
    )
    parser.add_argument(
        "--worldvqa-language",
        type=str,
        default=None,
        choices=["zh", "non-zh"],
        help="WorldVQA only: filter by language ('zh' or 'non-zh')",
    )
    parser.add_argument(
        "--evqa-data-path",
        type=str,
        default=None,
        help="EVQA only: local path to encyclopedic_vqa CSV (default: download from URL)",
    )
    parser.add_argument(
        "--evqa-instruction-override",
        type=str,
        default=None,
        help="EVQA only: replace per-example additional_instructions with this string. "
        "Used to standardize prompt across readers for fair comparison.",
    )

    # Required args
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-VL-4B-Instruct', 'gemini-3-pro-preview')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auto",
        help="Output JSONL path (default: auto-generate with model name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    # API args
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1", help="API base URL"
    )
    parser.add_argument("--api-key", type=str, default="dummy", help="API key")
    parser.add_argument(
        "--open-router",
        action="store_true",
        help="Use OpenRouter API (https://openrouter.ai). Requires --api-key or OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--commonstack",
        action="store_true",
        help="Use Commonstack API (https://api.commonstack.ai). Requires --api-key or COMMONSTACK_API_KEY env var.",
    )

    # General args
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to run (default: the whole filtered set)",
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Use SimpleQA Verified dataset instead of original SimpleQA dataset",
    )
    parser.add_argument(
        "--no-wiki-filter",
        action="store_true",
        help="Skip Wikipedia URL filter — include all examples (useful for API-based retrieval)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=200, help="Max concurrent requests"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (increase for tiled mode)",
    )
    parser.add_argument(
        "--screenshot-dir", type=str, default="screenshots", help="Screenshot directory"
    )

    # Retrieval mode args (mutually exclusive)
    parser.add_argument(
        "--url-screenshot",
        action="store_true",
        help="Use ground-truth screenshot for each example",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Max pixels for screenshot resize (for --url-screenshot). "
        "None=no resize (VLM handles it). "
        "Common values: 16777216 (16M, ~16K tokens), 4000000 (4M, ~4K tokens), 1000000 (1M, ~1K tokens)",
    )
    parser.add_argument(
        "--url-tiled-screenshot",
        action="store_true",
        help="Use ground-truth screenshot split into tiles",
    )
    parser.add_argument(
        "--url-text",
        action="store_true",
        help="Use text content from URL (crawl/jina/wikipedia)",
    )
    parser.add_argument(
        "--text-source",
        type=str,
        default="crawl",
        choices=["crawl", "jina", "wikipedia", "ds-serve"],
        help="Text source for --url-text or --text-vector: crawl (web scraping), jina (Jina Reader API), wikipedia (Wikipedia API), ds-serve (ds-serve API for external text augmentation)",
    )
    parser.add_argument(
        "--url-jina-reader",
        action="store_true",
        help="[DEPRECATED] Use --url-text --text-source jina instead",
    )
    parser.add_argument(
        "--retrieval-augment", action="store_true", help="Enable vector retrieval"
    )

    # Text RAG specific
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Max context chars (auto-calculated from --model-context-length if not set)",
    )
    parser.add_argument(
        "--model-context-length",
        type=int,
        default=65536,
        help="Model context length in tokens",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Reader max_tokens (generation budget)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Qwen3 thinking via chat_template_kwargs.enable_thinking=False",
    )
    parser.add_argument(
        "--text-cache",
        type=str,
        default="auto",
        help="Pre-fetched text JSONL (default: auto-generate based on text-source)",
    )

    # Vector retrieval specific
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=3,
        help="Top-k items the retriever fetches per query",
    )
    parser.add_argument(
        "--reader-top-k",
        type=int,
        default=None,
        help=(
            "Top-k items the reader actually sees per query. Defaults to --retrieval-top-k. "
            "Set lower than --retrieval-top-k to retrieve a larger superset once and downstream-evaluate "
            "k=1/2/3 from the same JSONL (full retrieved set is stored in `retrieved_images`). "
            "Per root CLAUDE.md the reader_top_k must be in {1, 2, 3}."
        ),
    )
    parser.add_argument(
        "--jina-api-key",
        type=str,
        default=os.environ.get("JINA_API_KEY"),
        help="Jina API key (defaults to the JINA_API_KEY env var)",
    )
    parser.add_argument(
        "--retrieval-cache", type=str, default=None, help="Embedding cache file"
    )
    parser.add_argument(
        "--single-vector", action="store_true", help="Use single vector mode"
    )

    # ColQwen2 LEANN retrieval args
    parser.add_argument(
        "--use-colqwen-retrieval",
        action="store_true",
        help="Use ColQwen2 LEANN retrieval instead of Jina API",
    )
    parser.add_argument(
        "--colqwen-index-path",
        type=str,
        default="./indexes/colqwen_screenshots.leann",
        help="Path to ColQwen2 LEANN index",
    )
    parser.add_argument(
        "--colqwen-model",
        type=str,
        default="colqwen2",
        choices=["colqwen2", "colqwen2.5", "colpali"],
        help="ColQwen2 model name",
    )
    parser.add_argument(
        "--colqwen-search-method",
        type=str,
        default="ann",
        choices=["ann", "exact", "exact-all"],
        help="ColQwen2 search method",
    )
    parser.add_argument(
        "--colqwen-first-stage-k",
        type=int,
        default=500,
        help="First stage k for ColQwen2 ANN search",
    )
    parser.add_argument(
        "--rebuild-colqwen-index",
        action="store_true",
        help="Rebuild ColQwen2 index even if it exists",
    )
    parser.add_argument(
        "--colqwen-recursive",
        action="store_true",
        help="Recursively search subdirectories when building ColQwen2 index",
    )

    # Qwen3-VL-Embedding retrieval args
    parser.add_argument(
        "--use-qwen3vl-embedding",
        action="store_true",
        help="Use Qwen3-VL-Embedding for tiled retrieval (single vector, 2048 dim)",
    )
    parser.add_argument(
        "--qwen3vl-model",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Qwen3-VL-Embedding model name",
    )
    parser.add_argument(
        "--qwen3vl-gpu-ids",
        type=str,
        default="2,3",
        help="Comma-separated GPU IDs for Qwen3-VL-Embedding (default: 2,3, TP=2; use 2,3,6,7 for TP=4 if P2P works)",
    )
    parser.add_argument(
        "--qwen3vl-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for Qwen3-VL-Embedding (default: 1)",
    )

    # Tiled vector retrieval args
    parser.add_argument(
        "--use-tiled-retrieval",
        action="store_true",
        help="Use tiled vector retrieval (splits images into fixed-size tiles)",
    )
    parser.add_argument(
        "--evqa-multimodal-query",
        action="store_true",
        help="EVQA only: pass text + image as separate modalities to query embedding (no query card). "
        "Uses GLDv2 landmark / iNaturalist image + question text. Requires --use-tiled-retrieval --use-qwen3vl-embedding.",
    )
    parser.add_argument(
        "--evqa-multimodal-query-text-only",
        action="store_true",
        help="EVQA ablation: with --evqa-multimodal-query, use text-only (no image) for query embedding.",
    )
    parser.add_argument(
        "--evqa-multimodal-query-image-only",
        action="store_true",
        help="EVQA ablation: with --evqa-multimodal-query, use image-only (no text) for query embedding.",
    )
    parser.add_argument(
        "--evqa-multi-image-query",
        action="store_true",
        help="EVQA only: use ALL query images per example for retrieval (not just the first). "
        "Each image is used for separate multimodal search, scores are aggregated via max. "
        "Requires --evqa-multimodal-query --use-tiled-retrieval --use-qwen3vl-embedding.",
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default=None,
        help="Directory to store image tiles (default: tiles-1024x{tile_height})",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=1024,
        help="Tile height in pixels (width is fixed to 1024)",
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=0, help="Overlap between tiles in pixels"
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Max tiles to use (auto-calculated from --model-context-length if not set)",
    )

    # Pixel query args
    parser.add_argument(
        "--pixel-query",
        action="store_true",
        help="Render queries as images (pixel queries) for retrieval and LLM input. "
        "Only works with --use-tiled-retrieval --use-qwen3vl-embedding.",
    )
    parser.add_argument(
        "--pixel-query-dir",
        type=str,
        default="pixel_queries",
        help="Directory to store rendered pixel query images (default: pixel_queries)",
    )

    parser.add_argument(
        "--prebuilt-tiles-dir",
        type=str,
        default=None,
        help="Path to a prebuilt tile directory (e.g. tiles-hard-mini/) containing ALL tiles "
        "(golden + distractor). Bypasses tile preparation — loads all .png files in the dir.",
    )

    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf", "biqwen3"],
        help="Backend for Qwen3-VL-Embedding: 'vllm' (default), 'hf' (HF direct GPU), or 'biqwen3' (BiQwen3 + optional PEFT adapter)",
    )
    parser.add_argument(
        "--peft-adapter",
        type=str,
        default=None,
        help="Path to PEFT/LoRA adapter checkpoint (only used with --embedding-backend biqwen3)",
    )

    # Pixel compression for generation (retrieval stays at original resolution)
    parser.add_argument(
        "--pixel-compress-ratio",
        type=float,
        default=None,
        help="Pixel compression ratio for generation images (float, ≥1.0). "
        "Divides total pixel count by this factor (dimensions by sqrt). "
        "E.g. for 1024x1024 tile: 1.5->837x837, 4->512x512, 9->341x341, 16->256x256, 25->205x205. "
        "Retrieval is always at original resolution. Default: no compression.",
    )

    # Local API retrieval
    parser.add_argument(
        "--local-api",
        action="store_true",
        help="Use local search API for tile retrieval (localhost:30888/search)",
    )
    parser.add_argument(
        "--local-api-url",
        type=str,
        default="http://localhost:30888/search",
        help="Local search API URL",
    )
    parser.add_argument(
        "--text-api",
        action="store_true",
        help="Use text search API for text chunk retrieval (text_search_api.py)",
    )
    parser.add_argument(
        "--text-api-url",
        type=str,
        default="http://localhost:30889/search",
        help="Text search API URL (default: http://localhost:30889/search)",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="Override FAISS nprobe for local API search (default: server default)",
    )
    parser.add_argument(
        "--no-query-image",
        action="store_true",
        help="Suppress attaching the example's query image to the retrieval query. "
        "Only affects --local-api (screenshot index): the retriever sends text-only. "
        "Reader still receives the query image. Useful for ablations that isolate the "
        "visual contribution of the retrieval query (not the reader).",
    )
    parser.add_argument(
        "--query-instruction",
        type=str,
        default=None,
        help="Override query embedding instruction string sent to the search API(s). "
        "Applies to --local-api (screenshot, :30888) and --text-api (:30889) and "
        "both legs of --hybrid. Default: server-side default "
        "('Retrieve images or text relevant to the user's query.' for screenshot, "
        "'Retrieve text relevant to the user's query.' for text).",
    )
    parser.add_argument(
        "--reader-extra-instructions",
        type=str,
        default=None,
        help="Extra free-form instructions appended to the reader's user-message "
        "additional_instructions (after the task's default, e.g. the short-answer "
        "directive for nq/triviaqa/nq_tables). Use for reader-side prompt ablations "
        "(e.g. visual-grid steering, few-shot format demos).",
    )
    parser.add_argument(
        "--reader-few-shot-json",
        type=str,
        default=None,
        help="Path to a JSON list of few-shot demos, each {'question','image_path','answer'}. "
        "When set, build_messages prepends (Example N, image, Q+A) blocks to every "
        "reader user-message. Works across pixel / text / naive modes.",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Use Qwen3-VL-Reranker to rerank retrieved tiles",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="Qwen/Qwen3-VL-Reranker-8B",
        help="Reranker model name (default: Qwen/Qwen3-VL-Reranker-8B)",
    )
    parser.add_argument(
        "--reranker-gpu-id",
        type=int,
        default=4,
        help="GPU ID for reranker (default: 4)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=3,
        help="Number of tiles to keep after reranking (default: 3)",
    )
    parser.add_argument(
        "--query-rewrite",
        action="store_true",
        help="Use LLM to rewrite questions into search queries before retrieval",
    )
    parser.add_argument(
        "--rewrite-model",
        type=str,
        default=None,
        help="Model for query rewriting (default: same as --model)",
    )
    parser.add_argument(
        "--rewrite-api-base",
        type=str,
        default=None,
        help="API base for rewrite model (default: same as --api-base)",
    )
    parser.add_argument(
        "--rewrite-api-key",
        type=str,
        default=None,
        help="API key for rewrite model (default: same as --api-key)",
    )

    # ReAct multi-turn retrieval
    parser.add_argument(
        "--react",
        action="store_true",
        help="Enable ReAct multi-turn retrieval: LLM can issue <search>query</search> to refine results",
    )
    parser.add_argument(
        "--react-max-turns",
        type=int,
        default=3,
        help="Maximum retrieval turns for ReAct (default: 3)",
    )
    parser.add_argument(
        "--react-prompt",
        type=str,
        default="v1",
        choices=["v1", "v2", "multihop"],
        help="ReAct prompt version: v1 (original), v2 (improved), or multihop (for multi-hop QA like 2wiki)",
    )

    # Text vector retrieval args (LEANN-based or ds-serve)
    parser.add_argument(
        "--text-vector",
        action="store_true",
        help="Use text vector retrieval with LEANN or ds-serve (if --text-source ds-serve)",
    )
    parser.add_argument(
        "--ds-serve-api-url",
        type=str,
        default="http://api.ds-serve.org:30888/search",
        help="ds-serve API URL (default: http://api.ds-serve.org:30888/search)",
    )
    parser.add_argument(
        "--text-embed-preset",
        type=str,
        default="qwen",
        choices=["qwen", "jina", "contriever"],
        help="Embedding preset: qwen (Qwen3-0.6B, default), jina (Jina API), or contriever (lightweight)",
    )
    parser.add_argument(
        "--rebuild-text-index",
        action="store_true",
        help="Force rebuild text index even if exists",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation (default: 32, lower if OOM)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Max tokens per chunk for text chunking (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Overlap tokens between chunks (default: 128)",
    )

    # Ablation A: OCR wrapper (image retrieve -> OCR -> text to reader)
    parser.add_argument(
        "--read-as-text-ocr",
        action="store_true",
        help="Ablation A: OCR retrieved tiles and feed text to reader. "
        "Requires an image retrieval mode (--local-api, --use-tiled-retrieval, etc.).",
    )
    parser.add_argument(
        "--ocr-url",
        type=str,
        default="http://localhost:8202/v1",
        help="OCR server base URL (OpenAI-compatible). Default: PaddleOCR-VL at :8202.",
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default="PaddlePaddle/PaddleOCR-VL",
        help="OCR model name passed to the chat completions request.",
    )
    parser.add_argument(
        "--ocr-cache",
        type=str,
        default="ocr_cache/paddleocr_vl.jsonl",
        help="JSONL cache for OCR results, keyed by image absolute path.",
    )
    parser.add_argument(
        "--ocr-concurrency",
        type=int,
        default=16,
        help="Max concurrent OCR requests to the server.",
    )

    # Ablation B: render text chunks as images (text retrieve -> rendered PNG -> VLM)
    parser.add_argument(
        "--render-as-image",
        action="store_true",
        help="Ablation B: render each retrieved text chunk as a compact Wikipedia-style image "
        "and feed images to the VLM reader. Requires --text-api.",
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="rendered_chunks",
        help="Directory for cached rendered chunk images.",
    )

    # Hybrid retrieval: merge image (LocalAPIRetriever) + text (TextAPIRetriever) hits
    # by raw normed-cosine score, take top-K overall, feed mixed-modality to reader.
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Hybrid retrieval: query both the image search API (--local-api-url) and the "
        "text search API (--text-api-url), merge hits by raw score desc, take top "
        "--retrieval-top-k overall. Feeds images for image hits and text for text hits "
        "to the same VL reader. Mutually exclusive with --local-api / --text-api / "
        "--read-as-text-ocr / --render-as-image.",
    )
    parser.add_argument(
        "--html-dom-lookup",
        action="store_true",
        help="HTML DOM lookup baseline: use text retrieval (--text-api-url) to find chunks, "
        "then look up their containing DOM structure in the original HTML from kiwix-serve. "
        "Returns structured HTML context (tables, sections) to the reader instead of flat text.",
    )
    parser.add_argument(
        "--llm-verify",
        action="store_true",
        help="(With --html-dom-lookup) Use an LLM (GPT-4.1-mini) to verify/improve DOM "
        "closure extraction. Falls back to heuristic when LLM call fails.",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    mode_count = sum(
        [
            args.url_screenshot,
            args.url_tiled_screenshot,
            args.url_text,
            args.url_jina_reader,
            args.retrieval_augment,
            args.use_tiled_retrieval,
            args.text_vector,
        ]
    )
    if mode_count > 1:
        print(
            "Error: Only one mode allowed: --url-screenshot, --url-tiled-screenshot, --url-text, --retrieval-augment, --use-tiled-retrieval, or --text-vector."
        )
        sys.exit(1)

    # Validate retrieval system selection
    if args.retrieval_augment and args.use_colqwen_retrieval and args.single_vector:
        print(
            "Warning: --single-vector is only for Jina API retrieval, ignoring for ColQwen2."
        )

    # Validate EVQA multimodal ablation flags
    if args.evqa_multimodal_query_text_only or args.evqa_multimodal_query_image_only:
        if not args.evqa_multimodal_query:
            print(
                "Error: --evqa-multimodal-query-text-only and --evqa-multimodal-query-image-only require --evqa-multimodal-query."
            )
            sys.exit(1)
        if (
            args.evqa_multimodal_query_text_only
            and args.evqa_multimodal_query_image_only
        ):
            print(
                "Error: --evqa-multimodal-query-text-only and --evqa-multimodal-query-image-only are mutually exclusive."
            )
            sys.exit(1)

    # Set default tiles-dir and screenshot-dir for EVQA (use cached paths)
    if args.task == "encyclopedic_vqa":
        if args.tiles_dir is None:
            args.tiles_dir = "tiles/evqa"
        if args.use_tiled_retrieval and args.screenshot_dir == "screenshots":
            args.screenshot_dir = "screenshots/evqa"
    elif args.tiles_dir is None:
        args.tiles_dir = f"tiles-1024x{args.tile_height}"

    # Auto-calculate max_context_chars if not set
    if args.max_context_chars is None:
        # Reserve tokens for: system prompt (~200), completion (2048), question (~200), buffer (~500)
        reserved_tokens = 3000
        available_tokens = args.model_context_length - reserved_tokens
        # Conservative estimate: ~2 chars per token (safe for mixed content)
        args.max_context_chars = available_tokens * 2
        logger.info(
            f"Auto-calculated max_context_chars: {args.max_context_chars} (from {args.model_context_length} context tokens)"
        )

    # Auto-generate text-cache path based on text-source
    if args.url_text and args.text_cache == "auto":
        cache_dir = "text_cache"
        os.makedirs(cache_dir, exist_ok=True)
        args.text_cache = os.path.join(
            cache_dir, f"text_cache_{args.text_source}.jsonl"
        )
        logger.info(f"Using text cache: {args.text_cache}")
    elif args.text_cache == "auto":
        args.text_cache = None  # Disable cache for non-text modes

    # Lower concurrency for heavy modes (web fetching, screenshots, retrieval, ds-serve, local-api)
    if args.local_api and args.max_concurrent > 5:
        logger.warning(
            f"Lowering max_concurrent from {args.max_concurrent} to 5 for local API stability."
        )
        args.max_concurrent = 5
    elif args.use_tiled_retrieval and args.max_concurrent > 10:
        logger.warning(
            f"Lowering max_concurrent from {args.max_concurrent} to 10 for tiled retrieval stability."
        )
        args.max_concurrent = 10
    elif (
        args.url_screenshot
        or args.url_text
        or args.url_jina_reader
        or args.retrieval_augment
        or (args.text_vector and args.text_source == "ds-serve")
    ) and args.max_concurrent > 20:
        reason = (
            "ds-serve API"
            if (args.text_vector and args.text_source == "ds-serve")
            else "image processing"
        )
        logger.warning(
            f"Lowering max_concurrent from {args.max_concurrent} to 20 for {reason} stability."
        )
        args.max_concurrent = 20

    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
