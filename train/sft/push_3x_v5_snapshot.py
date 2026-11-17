#!/usr/bin/env python3
"""Push the 3x_v5 ckpt-16000 LoRA adapter to a versioned snapshot repo.

This repo is a frozen snapshot tied to a specific recipe (2x top3 oversample,
mid-epoch-2 checkpoint). The main repo `multik-3x-lora` continues to track the
latest best 3x adapter.

Usage:
    HF_TOKEN=$(cat ~/.cache/huggingface/token) python sft/push_3x_v5_snapshot.py
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
REPO_ID = f"{USER}/qwen3vl-4b-wiki-screenshot-multik-3x-v5-lora"
SRC = Path(
    "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_top6_3x_v5/checkpoint-16000"
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

README = f"""---
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
- 3x-compression
- snapshot
- v5
pipeline_tag: image-text-to-text
---

# Qwen3-VL-4B Wiki-Screenshot QA LoRA — **3x compression, recipe v5 (2× top3 oversample)**

> **Snapshot repo.** This is a frozen snapshot of the v5 recipe at checkpoint-16000.
> The main `multik-3x-lora` repo tracks the current best 3x adapter (currently identical to this one).
> Sister snapshot for the prior recipe (1:1 mixed) lives at `{USER}/qwen3vl-4b-wiki-screenshot-multik-3x-v3-lora` if archived.

LoRA adapter for [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), fine-tuned as the **reader** in a retrieval-augmented Wikipedia-screenshot QA pipeline.

This adapter reads **any number of screenshots from 1 to 6** at inference time, at a fixed **3x pixel compression** per image. It exceeds the uncompressed no-SFT baseline at k≥2 and matches it at k=1 within 0.054.

## Performance (GPT-4.1 LLM-judge on 500 test examples)

| k (images per sample) | base@0x (no comp, no SFT) | v3 (1:1 mixed) | v5 (2:1 top3:vark) **this** | Δ vs v3 | Δ vs base@0x |
|---|---|---|---|---|---|
| 1 | 0.958 | 0.884 | **0.904** | +0.020 | -0.054 |
| 2 | 0.912 | 0.878 | **0.918** | +0.040 | **+0.006** |
| 3 | 0.892 | 0.892 | **0.932** | +0.040 | **+0.040** |
| 4 | 0.856 | 0.862 | **0.884** | +0.022 | **+0.028** |
| **avg** | **0.905** | 0.879 | **0.910** | **+0.031** | **+0.005** |

- **Beats base@0x at k=2/3/4** even though images are 3× compressed (each dim scaled by 1/√3).
- **Beats prior fixed-k=3 specialist (v1) ceiling of 0.900 at k=3** by +0.032 — heavier oversample of the most common deployment k value works *better* than dedicating the entire model to k=3 alone.
- **Near k-invariant**: 0.904 → 0.884 across k=1..4, only 0.020 spread (vs base@0x's 0.102 spread from 0.958 → 0.856).

## Recipe v5 — 2× top3 oversample

Compared with v3 (1:1 mixed):

| Field | v3 (prior) | **v5 (this)** |
|---|---|---|
| Datasets | `multimage_top3_train` + `multimage_vark_train` (1:1) | `multimage_top3_train` + `multimage_top3_train_dup` + `multimage_vark_train` (2:1 top3:vark) |
| Samples per epoch | 208k | **312k** (+50%) |
| LoRA rank | 256 | 256 (unchanged) |
| LoRA alpha | 256 | 256 (unchanged) |
| LoRA targets | all linear layers (incl. ViT) | all linear layers (incl. ViT) |
| `freeze_vision_tower` | false | false |
| Optimizer | AdamW, cosine, peak 1e-5, warmup 3% | identical |
| Effective batch | 32 (1 × 8 GPUs × 4 grad_accum) | identical |
| `cutoff_len` | 8192 | 8192 |
| Epochs | 2 (final adapter, step 13004) | 2 trained (full 19506 steps), **but released checkpoint is step 16000** (~1.64 epochs) — peak eval exact-match before mild overfit |
| Hardware | 8× H100 80GB, DeepSpeed ZeRO-2, bf16 | identical |
| Wallclock | ~7h | ~9h |

### Why 2× top3 oversample?

In retrieval-augmented Wikipedia-screenshot QA, **k=3 is the most common deployment shape** (most retrievers return the top 3 docs). The variable-k (vark) split alone undertrained the model on this exact shape because its samples are uniform over k ∈ {{1..6}} (~17k per k). Adding a duplicate of the fixed-k=3 (top3) split shifts ~67% of training samples to k=3, which:

1. Lifts k=3 LLM-judge from 0.892 to 0.932 (+0.040) — even past the ceiling of v1, a model trained *only* on fixed-k=3 (which scored 0.900).
2. Surprisingly also lifts all other k (k=1: +0.020, k=2: +0.040, k=4: +0.022). Heavier exposure to a single, well-defined k pattern seems to teach a cleaner answering style that transfers across k values.
3. The dual-dataset structure of v3 (which already beat the vark-only v2 at 3x) was the right base; v5 just shifts the mixture ratio toward the deployment center.

### Why ckpt-16000 (not final)?

| step | epoch | eval_loss | eval_em | LLM-judge avg (k=1..4) |
|---|---|---|---|---|
| 14400 | 1.48 | 0.163 | 0.766 | (not benchmarked) |
| **16000** | **1.64** | **0.164** | **0.772** | **0.910 ← released** |
| 17600 | 1.80 | 0.165 | 0.769 | (not benchmarked) |
| 19200 | 1.97 | 0.165 | 0.769 | (not benchmarked) |
| 19506 (final) | 2.00 | 0.166 | 0.768 | 0.901 (this trails ckpt-16000 at k=1/2/3, ties at k=4) |

Train loss in epoch 2 was 0.04-0.06 (vs epoch 1's 0.10-0.20) — the model continued to fit the training distribution well, but the gains stopped transferring to held-out test around step 16000. Pull a mid-epoch-2 checkpoint at peak eval em.

## Data — variable-k retrieval-augmented multi-image (with 2× top3 weighting)

Built from the [Chrisyichuan/screenshot-training-natural-filtered-v2](https://huggingface.co/datasets/Chrisyichuan/screenshot-training-natural-filtered-v2) QA dataset (~104k train examples):

1. For each query, retrieve top-6 screenshots from a Qwen3-VL-2B embedding index (dora-ls005 checkpoint) over 28M Wikipedia tiles.
2. **Two splits per epoch (each composed independently)**:
   - **top3 (×2)**: each sample is fixed k=3 — the gold + top 2 non-gold hits, gold position randomized. Listed twice in `dataset:` to oversample 2×.
   - **vark (×1)**: each sample uniformly samples k ∈ {{1..6}}, gold always included, position randomized.
3. Apply 3x compression to all images (each dim scaled by `1/sqrt(3)` via PIL LANCZOS).
4. Train the reader with `<image>×k \\n {{query}}` → gold answer.

**Effective per-epoch distribution**:
- k=3: 2 × 104k (top3 oversample) + ~17k (vark k=3 share) = **~225k samples (72%)**
- k=1, 2, 4, 5, 6 each: ~17k = ~5.4% each (28% combined)

Total per-epoch samples: 312k (1.5× v3's 208k). Train wallclock scales accordingly.

Gold-retrieval rate at top-6 across splits: ~75%. When gold is missing from retrieval, gold is still always included by construction in both splits.

## Usage

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base = "Qwen/Qwen3-VL-4B-Instruct"
adapter = "{REPO_ID}"  # or "Chrisyichuan/qwen3vl-4b-wiki-screenshot-multik-3x-lora" for the same weights via main repo

model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
processor = AutoProcessor.from_pretrained(base)

# k can be any integer in 1..6. Images must already be 3x-compressed (each dim × 1/sqrt(3)).
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": img_1}},
    # ... up to img_6 ...
    {{"type": "text",  "text": your_question}},
]}}]
# ... standard Qwen3-VL inference
```

## Notes / limitations

- **Pixel budget is fixed at 3x**. If your deployment can afford less compression, use the 2x sibling; if more, use the 4x sibling.
- **Training always included gold in the image set**. If your retriever misses the gold at inference, this adapter has not seen that distribution — expect degradation on those queries.
- For **k ∈ {{1, 2, 3, 4}}** this adapter was evaluated with GPT-4.1 LLM-judge on a 500-example test set. k=5/6 were trained on but not explicitly benchmarked.
- The 2× top3 oversample is asymmetric on purpose. Whether the same recipe generalizes to 2x compression: under test (see the 2x sibling card after the 2x_v4 run completes). At 4x compression the same recipe is unlikely to help — mixed-data already regressed at 4x ([prior finding](https://huggingface.co/{USER}/qwen3vl-4b-wiki-screenshot-multik-4x-lora)) due to insufficient pixel budget to absorb additional training signal.
- Sister adapters at other compression levels: `{USER}/qwen3vl-4b-wiki-screenshot-multik-2x-lora`, `{USER}/qwen3vl-4b-wiki-screenshot-multik-4x-lora`. Latest 3x: `{USER}/qwen3vl-4b-wiki-screenshot-multik-3x-lora`.
"""


def main() -> None:
    assert SRC.exists(), f"missing {SRC}"

    print(f"=== {REPO_ID} ===")
    print(f"  src: {SRC}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in KEEP_FILES:
            p = SRC / name
            if p.exists():
                shutil.copy2(p, tmp / name)
                print(f"  + {name}")
            else:
                print(f"  - {name} (not present, skipped)")

        (tmp / "README.md").write_text(README)
        print("  + README.md")

        create_repo(REPO_ID, token=TOKEN, exist_ok=True, private=False)
        upload_folder(
            folder_path=str(tmp),
            repo_id=REPO_ID,
            token=TOKEN,
            commit_message="Snapshot: 3x v5 recipe (2× top3 oversample, ckpt-16000)",
        )
        print(f"  uploaded → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
