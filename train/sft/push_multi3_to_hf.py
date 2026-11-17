#!/usr/bin/env python3
"""Push top-6-retrieval multi-image (3-image) LoRA adapters to HuggingFace.

Uploads adapter files + tokenizer + minimal metadata (no optimizer state,
no RNG, no DeepSpeed raw).

Usage:
    HF_TOKEN=$(cat ~/.cache/huggingface/token) python sft/push_multi3_to_hf.py
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

# Each entry: (compression label, adapter dir, LLM-judge score, config summary, epoch/step note)
BEST = [
    (
        "2x",
        "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_2x_v1",
        0.954,
        "r=256, 2ep, lr 1e-5, LLM+ViT LoRA, cutoff_len 5120, eff-batch 32 on 8× H100",
        "step 6502 (2 epochs)",
    ),
    (
        "3x",
        "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_3x_v1",
        0.900,
        "r=256, 2ep, lr 1e-5, LLM+ViT LoRA, cutoff_len 4096, eff-batch 32 on 8× H100",
        "step 6502 (2 epochs)",
    ),
    (
        "4x",
        "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_4x_v1",
        0.868,
        "r=256, 2ep, lr 1e-5, LLM+ViT LoRA, cutoff_len 3072, eff-batch 32 on 8× H100",
        "step 6502 (2 epochs)",
    ),
]

# Multi-image baselines (same 500 test set, GPT-4.1 judge, 3 images per sample, gold always in the 3)
BASE_MULTIIMAGE_0X = 0.892  # base Qwen3-VL-4B on uncompressed top-3
SINGLE_IMAGE_0X_BASELINE = (
    0.958  # base Qwen3-VL-4B on uncompressed single image (from RESULTS.md)
)

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


def build_readme(comp: str, judge: float, config: str, step_note: str) -> str:
    n_ratio = int(comp.rstrip("x"))
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
- retrieval-augmented
- compressed-images
- {comp}-compression
pipeline_tag: image-text-to-text
---

# Qwen3-VL-4B Multi-Image Wikipedia Screenshot QA LoRA — {comp} compression (top-3 retrieval)

LoRA adapter for [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) fine-tuned as the **reader** in a retrieval-augmented QA pipeline: given a question and **3 candidate Wikipedia screenshots** (one of which contains the answer, two are hard distractors), the model must locate the right image and extract the answer — all under **{comp} pixel compression**.

## Performance (GPT-4.1 LLM-judge on 500 test examples)

| Setup | LLM-judge |
|---|---|
| Uncompressed (0x) single-image, base Qwen3-VL-4B | {SINGLE_IMAGE_0X_BASELINE:.3f} |
| Uncompressed (0x) **multi-image (3)**, base Qwen3-VL-4B | {BASE_MULTIIMAGE_0X:.3f} |
| **This adapter @ {comp} multi-image (3)** | **{judge:.3f}** |

**Key observation:** {"The 2x multi-image SFT model nearly recovers the single-image uncompressed ceiling (0.958), and clearly exceeds the un-SFTed multi-image base (0.892) — fine-tuning compensates for both distractor confusion and 2x compression." if comp == "2x" else "Despite 3x pixel compression, this model slightly exceeds the un-SFTed multi-image uncompressed base (0.892) — fine-tuning compensates for both distractor confusion and 3x compression." if comp == "3x" else "At 4x compression the per-image pixel budget is ~¼ the original. Even with SFT, accuracy falls slightly below the un-SFTed multi-image 0x base (0.892). This checkpoint documents the compression–quality frontier at the extreme end; use 2x/3x for production."}

Gain over multi-image base @ 0x: **{judge - BASE_MULTIIMAGE_0X:+.3f}** ({100 * (judge - BASE_MULTIIMAGE_0X) / BASE_MULTIIMAGE_0X:+.1f}% relative).

## Training setup

- Method: LoRA ({config})
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (fork)
- Hardware: 8× H100 80GB, DeepSpeed ZeRO-2, bf16
- Checkpoint: {step_note}

## Data — retrieval-augmented multi-image

Training data was built from the [Chrisyichuan screenshot-training-natural-filtered-v2](https://huggingface.co/datasets/Chrisyichuan/screenshot-training-natural-filtered-v2) QA dataset:

1. For each query, retrieve top-6 screenshots from a Qwen3-VL-2B embedding index (dora-ls005 checkpoint) over 28M Wikipedia tiles.
2. Construct a 3-image set: **always include the gold** + up to 2 non-gold retrieved distractors.
3. Randomize gold position among the 3 to avoid positional shortcuts.
4. Apply **{comp} compression** (each dimension scaled by `1/sqrt({n_ratio})` via PIL LANCZOS).
5. Train the reader to answer the query given the 3 compressed images.

~104k training examples, 5.8k validation, 5.8k test. Gold-retrieval rate at top-6 across splits: ~75%.

## Usage

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base = "Qwen/Qwen3-VL-4B-Instruct"
adapter = "{USER}/qwen3vl-4b-wiki-screenshot-multi3-{comp}-lora"

model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
processor = AutoProcessor.from_pretrained(base)

# Three {comp}-compressed images (one gold + two distractors, any order)
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": img1}},
    {{"type": "image", "image": img2}},
    {{"type": "image", "image": img3}},
    {{"type": "text",  "text": your_question}},
]}}]
# ... standard Qwen3-VL inference
```

## Notes / limitations

- Training distribution **always includes the gold** in the 3-image set (by construction). If your retriever misses the gold, the model has not seen that distribution — expect degradation on those queries.
- Compression level is fixed at {comp}. Use the adapter that matches your deployment pixel budget.
- Sister adapters at other compression levels: {", ".join(f"`{USER}/qwen3vl-4b-wiki-screenshot-multi3-{x}-lora`" for x in ["2x", "3x", "4x"] if x != comp)}.
"""


def push_one(
    comp: str, adapter_dir: str, judge: float, config: str, step_note: str
) -> None:
    src = Path(adapter_dir)
    assert src.exists(), f"missing {src}"

    repo_id = f"{USER}/qwen3vl-4b-wiki-screenshot-multi3-{comp}-lora"
    print(f"\n=== {repo_id} ===")
    print(f"  src:  {src}")
    print(f"  judge: {judge}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in KEEP_FILES:
            p = src / name
            if p.exists():
                shutil.copy2(p, tmp / name)
                print(f"  + {name}")
            else:
                print(f"  - {name} (not present, skipped)")

        (tmp / "README.md").write_text(build_readme(comp, judge, config, step_note))
        print("  + README.md")

        create_repo(repo_id, token=TOKEN, exist_ok=True, private=False)
        upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            token=TOKEN,
            commit_message=f"Upload multi-image {comp} LoRA (LLM-judge {judge})",
        )
        print(f"  uploaded → https://huggingface.co/{repo_id}")


def main() -> None:
    for args in BEST:
        push_one(*args)
    print("\nAll done.")


if __name__ == "__main__":
    main()
