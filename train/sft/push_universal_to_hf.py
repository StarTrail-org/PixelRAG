#!/usr/bin/env python3
"""Push the mixed-llmvit universal adapter to HuggingFace."""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

TOKEN = os.environ["HF_TOKEN"]
USER = "Chrisyichuan"
REPO_ID = f"{USER}/qwen3vl-4b-wiki-screenshot-universal-lora"

SRC = Path(
    "/scratch/users/zwcolin/cxr_embeds/sft_output/qwen3vl_mixed_llmvit_v2/checkpoint-6503"
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
- compressed-images
- multi-compression
pipeline_tag: image-text-to-text
---

# Qwen3-VL-4B Wikipedia Screenshot QA — Universal Compression LoRA

A **single** LoRA adapter for [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) trained on images compressed at **four different levels simultaneously** (2x / 3x / 5x / 9x), so one adapter handles any of them at deployment.

## Performance (GPT-4.1 LLM-judge, 500 test examples)

| Compression | This universal adapter | Compression-specific adapter | Base Qwen (no SFT) |
|---|---|---|---|
| Uncompressed (0x) | — | — | 0.958 |
| **2x** | **0.940** | 0.948 | 0.904 |
| **3x** | **0.892** | 0.894 | 0.826 |
| **5x** | **0.692** | 0.730 | 0.554 |
| **9x** | **0.302** | 0.378 | 0.180 |

Trade-off: the universal adapter is **0.2–7.6 LLM-judge points** below each specialized adapter. At 3x it is essentially tied. Gap widens at 9x where compression-specific ViT tuning helps most.

## Training config

- Method: **LoRA r=256, α=256, target=all (LLM + ViT)** — `freeze_vision_tower: false`
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Checkpoint: step `6503` (1 epoch, ~5h on 4× H100)
- Framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (fork), DeepSpeed ZeRO-2, bf16
- Training data: 4 × 104k Wikipedia screenshot QA pairs, each compressed at 2x/3x/5x/9x → 416k mixed examples
- lr = 1e-5, cosine schedule, warmup_ratio 0.03, effective batch size 64

## Why train a universal adapter?

- **One LoRA to ship** instead of four.
- Compression-agnostic ViT: training sees the same text at multiple blur levels in the same run, regularizing the visual encoder toward blur-invariant features.
- At deployment, the caller just sends whatever compression suits their token budget — no routing logic.

## Usage

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base = "Qwen/Qwen3-VL-4B-Instruct"
adapter = "{USER}/qwen3vl-4b-wiki-screenshot-universal-lora"

model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
processor = AutoProcessor.from_pretrained(base)

# PIL image compressed at 2x / 3x / 5x / 9x — all work
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": your_compressed_image}},
    {{"type": "text",  "text": your_question}},
]}}]
# ... standard Qwen3-VL inference
```

## Data compression reference

Images were downscaled with PIL LANCZOS by `1/sqrt(N)` per dimension, so pixel count = `1/N` of original:
- 2x → 50% pixels
- 3x → 33% pixels
- 5x → 20% pixels
- 9x → 11% pixels

## Alternatives

If you need maximum accuracy at a known fixed compression, use the specialized adapters:
- [qwen3vl-4b-wiki-screenshot-2x-lora](https://huggingface.co/{USER}/qwen3vl-4b-wiki-screenshot-2x-lora)
- [qwen3vl-4b-wiki-screenshot-3x-lora](https://huggingface.co/{USER}/qwen3vl-4b-wiki-screenshot-3x-lora)
- [qwen3vl-4b-wiki-screenshot-5x-lora](https://huggingface.co/{USER}/qwen3vl-4b-wiki-screenshot-5x-lora)
- [qwen3vl-4b-wiki-screenshot-9x-lora](https://huggingface.co/{USER}/qwen3vl-4b-wiki-screenshot-9x-lora)
"""


def main() -> None:
    print(f"=== {REPO_ID} ===")
    print(f"  src: {SRC}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in KEEP_FILES:
            p = SRC / name
            if p.exists():
                shutil.copy2(p, tmp / name)
                print(f"  + {name}")
        (tmp / "README.md").write_text(README)
        print("  + README.md")
        create_repo(REPO_ID, token=TOKEN, exist_ok=True, private=False)
        upload_folder(
            folder_path=str(tmp),
            repo_id=REPO_ID,
            token=TOKEN,
            commit_message="Upload universal LoRA adapter (LLM+ViT on 2x/3x/5x/9x mixed, step 6503)",
        )
        print(f"  uploaded → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
