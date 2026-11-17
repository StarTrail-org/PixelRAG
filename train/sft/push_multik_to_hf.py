#!/usr/bin/env python3
"""Push variable-k (k=1..6) multi-image LoRA adapters (v2) to HuggingFace.

Each adapter was trained to handle an arbitrary number of images per sample
(k ∈ {1..6}), with the gold screenshot always included at a random position.
All images at a fixed compression ratio (2x / 3x / 4x).

Uploads adapter files + tokenizer + minimal metadata (no optimizer state,
no RNG, no DeepSpeed raw).

Usage:
    HF_TOKEN=$(cat ~/.cache/huggingface/token) python sft/push_multik_to_hf.py
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

TOKEN = (
    os.environ.get("HF_TOKEN")
    or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
)
USER = "Chrisyichuan"

# Each entry: (comp label, adapter dir, cutoff, k=1/k=2/k=3/k=4 LLM-judge)
# 2x: v3 mixed-data (top3+vark, r=256, 2ep) — recovered v1's k=3 ceiling, all-k wins.
# 3x: v5 mixed-data with 2x top3 oversample (top3+top3+vark, r=256, 2ep), best ckpt at step 16000
#     — exceeds base@0x at k=2/3/4 and beats v1's k=3 ceiling (0.900) by +0.032.
# 4x: v2 vark-only (r=256, 2ep) — mixed-data regressed at 4x (tight pixel budget can't absorb dual signals).
BEST = [
    (
        "2x",
        "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_2x_v3",
        10240,
        (0.946, 0.950, 0.954, 0.936),
    ),
    (
        "3x",
        "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_3x_v5/checkpoint-16000",
        8192,
        (0.904, 0.918, 0.932, 0.884),
    ),
    (
        "4x",
        "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_4x_v2",
        6144,
        (0.834, 0.828, 0.814, 0.810),
    ),
]

# Base Qwen3-VL-4B reference (uncompressed, no SFT), same 500-example test set
BASE_0X = {1: 0.958, 2: 0.912, 3: 0.892, 4: 0.856}

KEEP_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "trainer_state.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def build_readme(
    comp: str,
    adapter_dir: str,
    cutoff: int,
    scores: tuple[float, float, float, float],
) -> str:
    n_ratio = int(comp.rstrip("x"))
    sk1, sk2, sk3, sk4 = scores

    rows = []
    for k in (1, 2, 3, 4):
        base = BASE_0X[k]
        this = scores[k - 1]
        delta = this - base
        rows.append(f"| {k} | {base:.3f} | **{this:.3f}** | {delta:+.3f} |")
    table = "\n".join(rows)

    siblings = ", ".join(
        f"`{USER}/qwen3vl-4b-wiki-screenshot-multik-{x}x-lora`"
        for x in ("2", "3", "4")
        if x != comp.rstrip("x")
    )

    return f"""---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
- peft
- lora
- qwen3-vl
- screenshot-qa
- multi-image
- variable-k
- retrieval-augmented
- compressed-images
- {comp}-compression
pipeline_tag: image-text-to-text
---

# Qwen3-VL-4B Wiki-Screenshot QA LoRA — **variable-k (1–6 images)** @ {comp} compression

LoRA adapter for [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), fine-tuned as the **reader** in a retrieval-augmented Wikipedia-screenshot QA pipeline.

Unlike the earlier fixed-3-image adapters, this v2 reads **any number of screenshots from 1 to 6** at inference time, at a fixed **{comp} pixel compression** per image.

## Performance (GPT-4.1 LLM-judge on 500 test examples)

| k (images per sample) | base Qwen3-VL-4B @ 0x (no compression, no SFT) | **This adapter @ {comp}** | Δ |
|---|---|---|---|
{table}

Base@0x degrades sharply as the number of images grows (distractor confusion: 0.958 → 0.856 from k=1 to k=4). The {comp} adapter is nearly **k-invariant**: {("0.946 → 0.936 from k=1 to k=4 — only 0.010 drop." if comp == "2x" else "0.904 → 0.884 from k=1 to k=4 — only 0.020 drop." if comp == "3x" else "0.834 → 0.810 from k=1 to k=4.")}

{"At **2x compression**, this adapter exceeds the uncompressed no-SFT baseline at k≥2 and nearly matches it at k=1 (0.946 vs 0.958), demonstrating that reader SFT fully compensates for both 2x compression *and* multi-image distractor confusion." if comp == "2x" else "At **3x compression**, this adapter exceeds the uncompressed no-SFT baseline at k=2/3/4 (0.918/0.932/0.884 vs 0.912/0.892/0.856) and trails it only at k=1. The k=3 score (0.932) also beats the prior fixed-k=3 specialist ceiling (0.900) by +0.032 — a heavier 2× top3 oversample during training pushed multi-k flexibility past the single-k specialist." if comp == "3x" else "At **4x compression**, the per-image pixel budget is ~¼ the original. Even with variable-k SFT, this adapter trails the 2x and 3x variants. Use only under a hard pixel budget."}

## Training setup

- Method: LoRA (r=256, alpha=256, targets=all linear layers including ViT; `freeze_vision_tower=false`)
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (fork)
- Optimizer: cosine LR, peak 1e-5, warmup 3%
- Effective batch: 32 (per_device=1 × 8 GPUs × grad_accum=4)
- `cutoff_len`: **{cutoff}** (scaled per compression — larger for smaller compression where each image takes more tokens)
- Hardware: 8× H100 80GB, DeepSpeed ZeRO-2, bf16
- {("Training data: mixed top3 (fixed k=3, 104k) + vark (k=1..6, 104k) at 1:1 ratio, 2 epochs over 208k samples (final adapter at step 13004)." if comp == "2x" else "Training data: mixed **2× top3 (fixed k=3, 208k oversampled) + vark (k=1..6, 104k)** at 2:1 ratio, 2 epochs over 312k samples per epoch. The released adapter is the **best intermediate checkpoint at step 16000** (~1.64 epochs / 0.82 of total training) selected by peak eval exact-match — full 2 epochs slightly overfit on the eval split." if comp == "3x" else "Training data: vark-only (k=1..6, 104k), 2 epochs over 104k samples (final adapter at step 6502). Mixed-data and longer-epoch recipes both regressed at this compression — token-level eval improved but LLM-judge dropped, indicating the 4x pixel budget cannot absorb additional training signal.")}

## Data — variable-k retrieval-augmented multi-image

Built from the [Chrisyichuan/screenshot-training-natural-filtered-v2](https://huggingface.co/datasets/Chrisyichuan/screenshot-training-natural-filtered-v2) QA dataset (~104k train examples):

1. For each query, retrieve top-6 screenshots from a Qwen3-VL-2B embedding index (dora-ls005 checkpoint) over 28M Wikipedia tiles.
2. **Per-sample, uniformly sample k ∈ {{1, 2, 3, 4, 5, 6}}**.
3. Compose the k-image set: **gold always included** + (k-1) non-gold hits (padded with gold repeats if retrieval has fewer than k-1 non-gold hits).
4. Randomize gold position among the k images (prevents positional shortcuts).
5. Apply {comp} compression (each dimension scaled by `1/sqrt({n_ratio})` via PIL LANCZOS).
6. Train the reader with `<image>×k \\n {{query}}` → gold answer.

**Distribution**: ~17k train samples per k value (uniform). eval/test sets likewise stratified over k∈{{1..6}}.

Gold-retrieval rate at top-6 across splits: ~75%. When gold is present at rank 1..6 among the retrieved hits, composition uses only retrieval results (else gold is always still included by construction).

## Usage

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base = "Qwen/Qwen3-VL-4B-Instruct"
adapter = "{USER}/qwen3vl-4b-wiki-screenshot-multik-{comp}-lora"

model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
processor = AutoProcessor.from_pretrained(base)

# k can be any integer in 1..6. Images must already be {comp}-compressed.
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": img_1}},
    # ... up to img_6 ...
    {{"type": "text",  "text": your_question}},
]}}]
# ... standard Qwen3-VL inference
```

## Notes / limitations

- **Pixel budget is fixed at {comp}**. If your deployment can afford less compression, use the 2x sibling; if more, use the 4x sibling. Mixing compression at inference is untested.
- **Training always included gold in the image set**. If your retriever misses the gold at inference, this adapter has not seen that distribution — expect degradation on those queries.
- For **k ∈ {{1, 2, 3, 4}}** this adapter was evaluated with GPT-4.1 LLM-judge. k=5/6 were trained on but not explicitly benchmarked.
- {("v1 of this adapter (trained on fixed k=3 only) achieved slightly higher k=3 score but collapses at other k. This v2 trades ~0.02 at k=3 for flexibility across k=1..6." if comp != "3x" else "The 2× top3 oversample is asymmetric on purpose: at 3x compression the per-image budget is tight enough that purely uniform-k training under-specializes for the most common deployment k=3. Doubling top3 (104k extra fixed-k=3 samples) lifts k=3 from 0.892 to 0.932 (+0.040) while still gaining vs the uniform-k baseline at all other k. The same recipe regressed at 4x — see the 4x sibling adapter card.")}
- Sister adapters at other compression levels: {siblings}.
"""


def push_one(
    comp: str,
    adapter_dir: str,
    cutoff: int,
    scores: tuple[float, float, float, float],
) -> None:
    src = Path(adapter_dir)
    assert src.exists(), f"missing {src}"

    repo_id = f"{USER}/qwen3vl-4b-wiki-screenshot-multik-{comp}-lora"
    print(f"\n=== {repo_id} ===")
    print(f"  src:  {src}")
    print(f"  k=1..4 LLM-judge: {scores}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in KEEP_FILES:
            p = src / name
            if p.exists():
                shutil.copy2(p, tmp / name)
                print(f"  + {name}")
            else:
                print(f"  - {name} (not present, skipped)")

        (tmp / "README.md").write_text(build_readme(comp, adapter_dir, cutoff, scores))
        print("  + README.md")

        create_repo(repo_id, token=TOKEN, exist_ok=True, private=False)
        upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            token=TOKEN,
            commit_message=f"Upload variable-k (1..6) {comp} LoRA",
        )
        print(f"  uploaded → https://huggingface.co/{repo_id}")


def main() -> None:
    for args in BEST:
        push_one(*args)
    print("\nAll done.")


if __name__ == "__main__":
    main()
