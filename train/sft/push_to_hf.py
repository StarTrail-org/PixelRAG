#!/usr/bin/env python3
"""Push best LoRA adapters from each compression level to HuggingFace.

Uploads only adapter files + tokenizer + minimal metadata (no optimizer state,
no RNG, no DeepSpeed raw).
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

TOKEN = os.environ["HF_TOKEN"]
USER = "Chrisyichuan"

# (compression, run_dir, step, llm_judge, config_summary)
# NEW: LLM+ViT LoRA (freeze_vision_tower: false) for 3x/5x/9x. 2x tied with old LLM-only.
BEST = [
    # Universal adapter: one LoRA across all compression levels (2x/3x/5x/9x).
    (
        "universal",
        "qwen3vl_mixed_llmvit_v1",
        6503,
        0.920,
        "r=128, 1ep, lr 2e-5, ViT unfrozen, trained on 2x+3x+5x+9x mixed data",
    ),
]

BASE_BASELINE = {
    "2x": 0.904,
    "3x": 0.826,
    "5x": 0.554,
    "9x": 0.180,
    "universal": (0.904 + 0.826 + 0.554 + 0.180) / 4,
}

BASE_CKPT_DIR = "/scratch/users/zwcolin/cxr_embeds/sft_output"

# Files to KEEP when uploading (strip optimizer state / RNG / zero_to_fp32)
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
    comp: str, step: int, judge: float, config: str, base_judge: float
) -> str:
    return f"""---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
- peft
- lora
- qwen3-vl
- screenshot-qa
- compressed-images
- {comp}-compression
pipeline_tag: image-text-to-text
---

# Qwen3-VL-4B Wikipedia Screenshot QA LoRA — {comp} compression

LoRA adapter for [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) fine-tuned to answer natural-language questions about Wikipedia-screenshot chunks, specifically on images compressed by **{comp}** (each dim scaled by 1/√{comp[:-1]}).

## Performance (GPT-4.1 LLM-judge on 500 test examples)

| Setup | LLM-judge |
|---|---|
| Uncompressed (0x) ceiling, base Qwen3-VL-4B | 0.958 |
| **This adapter @ {comp}** | **{judge:.3f}** |
| Base Qwen3-VL-4B @ {comp} (no SFT) | {base_judge:.3f} |

SFT gain over base at {comp}: **+{judge - base_judge:.3f}** ({100 * (judge - base_judge) / base_judge:.1f}% relative).

## Training config

- Method: LoRA ({config})
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Checkpoint: step `{step}`
- Framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (fork)
- Trained on 4× H100 80GB (DeepSpeed ZeRO-2, bf16)
- Dataset: Wikipedia screenshot-QA pairs compressed with PIL LANCZOS

## Data preparation

Training images were downscaled by `1/sqrt({comp[:-1]})` per dimension using PIL LANCZOS, e.g. a 1200×800 screenshot becomes {int(1200 / int(comp[:-1]) ** 0.5)}×{int(800 / int(comp[:-1]) ** 0.5)} px (~{100 / int(comp[:-1]):.0f}% of original pixels).

## Usage

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base = "Qwen/Qwen3-VL-4B-Instruct"
adapter = "{USER}/qwen3vl-4b-wiki-screenshot-{comp}-lora"

model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
processor = AutoProcessor.from_pretrained(base)

# PIL image already compressed to {comp}
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": your_compressed_image}},
    {{"type": "text",  "text": your_question}},
]}}]
# ... standard Qwen3-VL inference
```

## Notes / limitations

- The adapter is specific to the **{comp} compression level** and does not necessarily generalize to higher or lower compression. Use the adapter whose level matches your deployment.
- At {comp}, SFT recovers {100 * (judge - base_judge) / (0.958 - base_judge):.0f}% of the compression-induced accuracy drop relative to uncompressed Qwen3-VL-4B.
- See the full experiment matrix and findings in `sft/RESULTS.md` of the source repo.
"""


def push_one(comp: str, run_dir: str, step: int, judge: float, config: str) -> None:
    src = Path(BASE_CKPT_DIR) / run_dir / f"checkpoint-{step}"
    assert src.exists(), f"missing {src}"

    repo_id = f"{USER}/qwen3vl-4b-wiki-screenshot-{comp}-lora"
    print(f"\n=== {repo_id} ===")
    print(f"  src:  {src}")
    print(f"  step: {step}, judge: {judge}")

    # Stage files in a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in KEEP_FILES:
            p = src / name
            if p.exists():
                shutil.copy2(p, tmp / name)
                print(f"  + {name}")
            else:
                print(f"  - {name} (not present, skipped)")

        # Write README
        base_judge = BASE_BASELINE[comp]
        (tmp / "README.md").write_text(
            build_readme(comp, step, judge, config, base_judge)
        )
        print("  + README.md")

        # Create repo + upload
        create_repo(repo_id, token=TOKEN, exist_ok=True, private=False)
        upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            token=TOKEN,
            commit_message=f"Upload {comp} LoRA adapter (step {step}, LLM-judge {judge})",
        )
        print(f"  uploaded → https://huggingface.co/{repo_id}")


def main() -> None:
    for args in BEST:
        push_one(*args)
    print("\nAll done.")


if __name__ == "__main__":
    main()
