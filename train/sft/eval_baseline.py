#!/usr/bin/env python3
"""Eval Qwen3-VL-4B (base or SFT-LoRA) on test set at a given compression level.

Reports three metrics:
  - exact_match          : predicted.lower() == golden.lower()
  - char_accuracy        : character-level fuzzy match (weak, kept for backcompat)
  - llm_judge_accuracy   : GPT-4.1 grade using the SimpleQA-style grader template
                           (A=correct / B=incorrect / C=not_attempted)

Environment:
  OPENAI_API_KEY, OPENAI_BASE_URL must be set. Source .env at repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class EvalResult(TypedDict, total=False):
    query: str
    golden: str
    predicted: str
    chunk_path: str
    image_missing: bool
    n_images: int
    judge_grade: str
    judge_correct: bool


class EMCharMetrics(TypedDict):
    exact_match: float
    char_accuracy: float
    scored: int


class JudgeMetrics(TypedDict):
    llm_judge_accuracy: float
    llm_judge_correct: int
    llm_judge_total: int


# Reused from train_contrastors.py — SimpleQA-style grader, returns A/B/C.
_GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.
For numerical answers the predicted answer must be correct to the last significant figure in the gold answer. The gold target may contain more information than the question; the predicted answer only needs to contain what the question asks.
Do not punish typos in names if it is clearly the same name.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself; we are just grading the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it."""


def _resolve_image_path(ex: dict[str, str], images_root: str) -> str:
    """chunk_path is relative to the dataset root (e.g. images/shard_000/...).
    images_root is the directory that contains the `images/` subtree (compressed or original)."""
    rel = ex["chunk_path"]
    if os.path.isabs(rel):
        return rel
    return os.path.join(images_root, rel)


def run_inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    examples: list[dict[str, str]],
    images_root: str,
    device: str,
    desc: str,
    max_new_tokens: int = 128,
    enable_thinking: bool = False,
) -> list[EvalResult]:
    """Run VQA inference on a list of examples; returns list of (golden, predicted) pairs."""
    results = []
    for ex in tqdm(examples, desc=desc):
        img_path = _resolve_image_path(ex, images_root)
        if not os.path.exists(img_path):
            results.append(
                {
                    "query": ex["query"],
                    "golden": ex["answer"].strip(),
                    "predicted": "",
                    "image_missing": True,
                }
            )
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": ex["query"]},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        gen_ids = out_ids[0][inputs.input_ids.shape[1] :]
        pred = processor.decode(gen_ids, skip_special_tokens=True).strip()

        results.append(
            {
                "query": ex["query"],
                "golden": ex["answer"].strip(),
                "predicted": pred,
                "chunk_path": ex["chunk_path"],
            }
        )
    return results


def compute_em_char(results: list[EvalResult]) -> EMCharMetrics:
    correct_em = 0
    char_correct = 0
    char_total = 0
    scored = 0
    for r in results:
        if r.get("image_missing"):
            continue
        scored += 1
        pred = r["predicted"].lower()
        gold = r["golden"].lower()
        if pred == gold:
            correct_em += 1
        if gold or pred:
            matches = sum(1 for a, b in zip(pred, gold) if a == b)
            char_correct += matches
            char_total += max(len(pred), len(gold))
    return {
        "exact_match": correct_em / scored if scored else 0.0,
        "char_accuracy": char_correct / char_total if char_total else 0.0,
        "scored": scored,
    }


def grade_with_gpt(
    results: list[EvalResult], model: str, concurrency: int = 16
) -> JudgeMetrics:
    """Grade predictions with GPT-4.1. Returns list of (bool correct, raw grade)."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY + OPENAI_BASE_URL from env

    def _grade(idx, r):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": _GRADER_TEMPLATE.format(
                            question=r["query"],
                            target=r["golden"],
                            predicted_answer=r["predicted"] or "(no answer)",
                        ),
                    }
                ],
                max_tokens=5,
                temperature=0,
            )
            grade = resp.choices[0].message.content.strip()
            # A = CORRECT
            is_correct = bool(re.match(r"^\s*A\b", grade))
            return idx, is_correct, grade
        except Exception as e:
            return idx, False, f"ERR:{e}"

    scored = [(i, r) for i, r in enumerate(results) if not r.get("image_missing")]
    n = len(scored)
    verdicts = [("", False)] * len(results)
    correct = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_grade, i, r) for i, r in scored]
        for fut in tqdm(as_completed(futures), total=n, desc="GPT judge"):
            idx, is_correct, grade = fut.result()
            verdicts[idx] = (grade, is_correct)
            if is_correct:
                correct += 1

    for i, (grade, is_correct) in enumerate(verdicts):
        results[i]["judge_grade"] = grade
        results[i]["judge_correct"] = is_correct
    return {
        "llm_judge_accuracy": correct / n if n else 0.0,
        "llm_judge_correct": correct,
        "llm_judge_total": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter checkpoint (optional). If unset, use base model.",
    )
    p.add_argument(
        "--dataset-dir",
        default="/scratch/users/zwcolin/cxr_embeds/external_data/screenshot-training-natural-filtered-v2",
        help="Directory containing <split>_hn_with_answer.jsonl",
    )
    p.add_argument("--split", default="test", choices=["train", "eval", "test"])
    p.add_argument(
        "--images-root",
        default=None,
        help="Directory that contains the images/ subtree. "
        "Default: --dataset-dir (uncompressed). For compressed eval, pass "
        "/scratch/users/zwcolin/cxr_embeds/sft_data/compressed_Nx",
    )
    p.add_argument("--n-examples", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable <think></think> mode. Auto-bumps max-new-tokens to 512 if still default.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--judge",
        action="store_true",
        default=True,
        help="Run GPT-4.1 judge (default on).",
    )
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--judge-model", default="gpt-4.1-2025-04-14")
    p.add_argument("--judge-concurrency", type=int, default=16)
    p.add_argument(
        "--tag",
        required=True,
        help="Run label (e.g. 'base_0x', 'sft_3x'). Used in output filename.",
    )
    p.add_argument(
        "--output-dir",
        default="/scratch/users/zwcolin/cxr_embeds/cxr_embedding/sft/eval_out",
    )
    args = p.parse_args()

    if args.judge and not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: --judge set but OPENAI_API_KEY not in env. "
            "Run `source .env` (repo root) or pass --no-judge.",
            file=sys.stderr,
        )
        sys.exit(1)

    images_root = args.images_root or args.dataset_dir

    # Load model (+ optional adapter)
    print(f"Loading base model: {args.model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    if args.adapter:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model)

    # Load examples
    jsonl = os.path.join(args.dataset_dir, f"{args.split}_hn_with_answer.jsonl")
    with open(jsonl) as f:
        examples = [json.loads(line) for line in f]
    if args.n_examples > 0:
        examples = examples[: args.n_examples]
    print(f"Loaded {len(examples)} examples from {jsonl}")
    print(f"Images root: {images_root}")

    # Auto-bump max-new-tokens for thinking mode
    if args.thinking and args.max_new_tokens < 512:
        args.max_new_tokens = 512
        print("Thinking enabled: bumped max_new_tokens to 512")

    # Inference
    results = run_inference(
        model,
        processor,
        examples,
        images_root,
        args.device,
        desc=f"eval[{args.tag}]",
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
    )

    # Compute metrics
    metrics = compute_em_char(results)
    print(f"\n=== {args.tag} ===")
    print(f"  scored:        {metrics['scored']} / {len(results)}")
    print(f"  exact_match:   {metrics['exact_match']:.4f}")
    print(f"  char_accuracy: {metrics['char_accuracy']:.4f}")

    if args.judge:
        judge_metrics = grade_with_gpt(
            results, args.judge_model, args.judge_concurrency
        )
        metrics.update(judge_metrics)
        print(
            f"  llm_judge:     {metrics['llm_judge_accuracy']:.4f} "
            f"({metrics['llm_judge_correct']}/{metrics['llm_judge_total']})"
        )

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "dataset_dir": args.dataset_dir,
        "images_root": images_root,
        "split": args.split,
        "n_examples": len(results),
        "judge_model": args.judge_model if args.judge else None,
        "metrics": metrics,
        "results": results,
    }
    fname = f"eval_{args.tag}_{args.split}_n{len(results)}.json"
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {fpath}")


if __name__ == "__main__":
    main()
