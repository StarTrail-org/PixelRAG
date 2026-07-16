"""Self-contained LLM-as-judge grader for the PixelRAG reproduction.

Implements the paper's evaluation path (encyclopedic_vqa / mmsearch / worldvqa) as a
standalone module. Behaviour is faithful to the paper grader:

- Judge prompt = JUDGE_WORLDQA_PROMPT_EN (verbatim from MoonshotAI/WorldVQA),
  loaded from eval/repro_assets/judge_worldvqa_prompt.txt.
- Ground truth:
    * encyclopedic_vqa -> "Any of: " + " | ".join(reference_list)  (ANY match = correct)
    * mmsearch / worldvqa -> gt_answer (single string)
- The model response has <think>...</think> stripped before judging.
- Judge model gpt-4.1-2025-04-14, temperature=0; verdict parsed from a
  `Label: Correct|Incorrect|Unattempted` line.
- score = #Correct / N.

CLI:
    python -m lib.grader <task> <responses.jsonl> [--grader-model gpt-4.1-2025-04-14]
Requires OPENAI_API_KEY (+ optional OPENAI_BASE_URL) in the environment.
"""

import argparse
import asyncio
import json
import os
import re
import string
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent / "repro_assets"
JUDGE_WORLDQA_PROMPT_EN = (_ASSETS / "judge_worldvqa_prompt.txt").read_text()
SIMPLEQA_GRADER_TEMPLATE = (_ASSETS / "simpleqa_grader_template.txt").read_text()

# Which grader each task uses (matches paper scripts/evaluate.py dispatch).
WORLDVQA_TASKS = {
    "encyclopedic_vqa",
    "mmsearch",
    "worldvqa",
    "factualvqa",
    "webqa",
    "multimodalqa",
}
EXACT_MATCH_TASKS = {"nq", "nq_tables", "triviaqa"}
SIMPLEQA_TASKS = {"simpleqa", "simpleqa_verified"}

DEFAULT_GRADER_MODEL = "gpt-4.1-2025-04-14"
# Match the paper grader sampler (scripts/evaluate.py -> ChatCompletionSampler):
# system message "You are a helpful assistant.", temperature=0, max_tokens=1000, seed=42.
GRADER_SYSTEM_MESSAGE = "You are a helpful assistant."
GRADER_MAX_TOKENS = 1000
GRADER_SEED = 42


def strip_think(text: str) -> str:
    # Verbatim from paper worldvqa_eval.strip_think_tags.
    if text is None:
        return ""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    elif "think>" in text:
        return text.split("think>")[-1].strip()
    return text


def build_ground_truth(task: str, original_data: dict) -> str:
    """Match evaluate.py convert_to_evaluate_format."""
    if task in EXACT_MATCH_TASKS:
        # nq / nq_tables / triviaqa carry a list of acceptable gold spans/aliases.
        golds = [str(g) for g in (_golds_for(task, original_data) or []) if g]
        return "Any of: " + " | ".join(golds) if golds else ""
    if task == "encyclopedic_vqa":
        refs = original_data.get("reference_list") or []
        if refs:
            return "Any of: " + " | ".join(refs)
        return original_data.get("answer", "") or original_data.get("gt_answer", "")
    # mmsearch / worldvqa / simplevqa / factualvqa
    return original_data.get("gt_answer", "") or original_data.get("answer", "")


def parse_label(judge_text: str) -> str:
    m = re.search(
        r"Label:\s*(Correct|Incorrect|Unattempted)", judge_text, re.IGNORECASE
    )
    if m:
        return m.group(1).lower()
    tl = judge_text.lower()
    if "incorrect" in tl:
        return "incorrect"
    if "unattempted" in tl:
        return "unattempted"
    if "correct" in tl:
        return "correct"
    return "incorrect"


# ---------------------------------------------------------------------------
# NQ / NQ-Tables exact-match (verbatim from short_form_qa_eval.short_form_eval)
# ---------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    s = re.sub(
        r"\b(a|an|the)\b",
        " ",
        s.lower().translate(str.maketrans("", "", string.punctuation)),
    )
    return " ".join(s.split())


def is_exact_match(prediction: str, golds) -> bool:
    prediction = (prediction or "").replace("Exact Answer: ", "").strip()
    pred_norm = _normalize_text(prediction)
    return any(_normalize_text(str(g)) == pred_norm for g in golds)


def _golds_for(task: str, od: dict):
    if task in EXACT_MATCH_TASKS:
        g = (
            od.get("answers")
            or od.get("gold_answers")
            or od.get("reference_list")
            or od.get("answer")
            or od.get("gt_answer")
        )
        return g if isinstance(g, list) else [g]
    return None


def grade_exact_match(path: str) -> dict:
    rows = [json.loads(l) for l in open(path)]
    c = 0
    for d in rows:
        golds = _golds_for("nq", d.get("original_data", {}))
        if is_exact_match(strip_think(d.get("final_response")), golds):
            c += 1
    n = len(rows)
    return {
        "task": "exact_match",
        "file": path,
        "n": n,
        "correct": c,
        "incorrect": n - c,
        "unattempted": 0,
        "errors": 0,
        "score": c / n if n else 0.0,
    }


async def grade_file(
    task: str,
    path: str,
    grader_model: str = DEFAULT_GRADER_MODEL,
    concurrency: int = 16,
    llm_judge: bool = False,
) -> dict:
    # nq/nq_tables/triviaqa default to strict exact-match (no API key needed). The paper's
    # published numbers for these used the LLM judge (semantic match) — pass --llm-judge.
    if task in EXACT_MATCH_TASKS and not llm_judge:
        return grade_exact_match(path)
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL")
    )
    rows = [json.loads(l) for l in open(path)]
    sem = asyncio.Semaphore(concurrency)
    labels = [None] * len(rows)

    is_sqa = task in SIMPLEQA_TASKS

    async def judge(i, d):
        od = d.get("original_data", {})
        answer = strip_think(d.get("final_response"))
        if is_sqa:
            target = od.get("answer", "") or od.get("gt_answer", "")
            prompt = SIMPLEQA_GRADER_TEMPLATE.format(
                question=d.get("problem", ""), target=target, predicted_answer=answer
            )
        else:
            gt = build_ground_truth(task, od)
            prompt = JUDGE_WORLDQA_PROMPT_EN.format(
                question=d.get("problem", ""),
                model_answer=answer,
                ground_truth_answer=gt,
            )
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=grader_model,
                    temperature=0,
                    max_tokens=GRADER_MAX_TOKENS,
                    seed=GRADER_SEED,
                    messages=[
                        {"role": "system", "content": GRADER_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                )
                out = r.choices[0].message.content
                if is_sqa:
                    m = re.search(r"(A|B|C)", out or "")
                    letter = m.group(0) if m else "C"
                    labels[i] = {"A": "correct", "B": "incorrect", "C": "unattempted"}[
                        letter
                    ]
                else:
                    labels[i] = parse_label(out)
            except Exception as e:
                labels[i] = ("__error__", str(e))

    await asyncio.gather(*[judge(i, d) for i, d in enumerate(rows)])
    errs = [l for l in labels if isinstance(l, tuple)]
    verdicts = [l for l in labels if isinstance(l, str)]
    n = len(verdicts)
    c = verdicts.count("correct")
    inc = verdicts.count("incorrect")
    una = verdicts.count("unattempted")
    return {
        "task": task,
        "file": path,
        "n": n,
        "correct": c,
        "incorrect": inc,
        "unattempted": una,
        "errors": len(errs),
        "score": c / n if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="encyclopedic_vqa | mmsearch | worldvqa | ...")
    ap.add_argument("jsonl", help="responses jsonl from run_bench.py")
    ap.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--llm-judge",
        action="store_true",
        help="For nq/nq_tables/triviaqa: grade with the gpt-4.1 LLM judge (semantic "
        "match — what the paper used for its published numbers) instead of strict "
        "exact-match. Requires OPENAI_API_KEY.",
    )
    args = ap.parse_args()
    res = asyncio.run(
        grade_file(
            args.task, args.jsonl, args.grader_model, args.concurrency, args.llm_judge
        )
    )
    mode = (
        "LLM-judge"
        if args.task in EXACT_MATCH_TASKS and args.llm_judge
        else ("exact-match" if args.task in EXACT_MATCH_TASKS else "LLM-judge")
    )
    print(
        f"{Path(res['file']).name} [{mode}]: {res['correct']}/{res['n']} = {res['score']:.4f} "
        f"(C={res['correct']} I={res['incorrect']} U={res['unattempted']} err={res['errors']})"
    )
    print(f"Score: {res['score']:.3f}")


if __name__ == "__main__":
    main()
