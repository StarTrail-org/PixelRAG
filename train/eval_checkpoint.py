"""Standalone checkpoint eval using train_contrastors.py functions."""

import argparse
import json
import os

import torch
from models.biqwen3 import BiQwen3
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint or 'base'")
    parser.add_argument("--test-data", default="training/data/test_miniv6.json")
    parser.add_argument("--vllm-url", default="http://localhost:8201/v1")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--grader-model", default="gpt-4.1-2025-04-14")
    parser.add_argument("--max-num-visual-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load model
    model = BiQwen3.from_pretrained(
        "Qwen/Qwen3-VL-Embedding-2B", dtype=torch.bfloat16
    ).to(device)
    if args.checkpoint != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.checkpoint)
        print(f"LoRA loaded: {args.checkpoint}")
    model.eval()

    # Processor with left padding + visual token config
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")
    processor.tokenizer.padding_side = "left"
    ppt = (
        processor.image_processor.patch_size**2
        * processor.image_processor.merge_size**2
    )
    processor.image_processor.max_pixels = args.max_num_visual_tokens * ppt
    processor.image_processor.min_pixels = max(
        processor.image_processor.min_pixels, ppt
    )
    processor.image_processor.size["longest_edge"] = (
        processor.image_processor.max_pixels
    )
    processor.image_processor.size["shortest_edge"] = (
        processor.image_processor.min_pixels
    )

    # Init chat templates
    import train_contrastors as tc

    tc.init_chat_templates(processor)

    # Load test data
    with open(args.test_data) as f:
        td = json.load(f)
    tiles_dir = td["tiles_dir"]
    doc_paths = sorted(
        [
            os.path.join(tiles_dir, f)
            for f in os.listdir(tiles_dir)
            if f.endswith(".png")
        ]
    )
    test_data = {
        "questions": td["questions"],
        "doc_paths": doc_paths,
        "golden_mapping": td["golden_mapping"],
    }
    print(f"Test: {len(td['questions'])} queries, {len(doc_paths)} tiles")

    # Run eval
    output_path = None
    if args.checkpoint != "base":
        ckpt_name = (
            os.path.basename(os.path.dirname(args.checkpoint))
            + "_"
            + os.path.basename(args.checkpoint)
        )
    else:
        ckpt_name = "base"
    test_name = os.path.splitext(os.path.basename(args.test_data))[0]
    output_path = f"training/eval_results/{ckpt_name}_{test_name}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = tc.run_miniv6_eval(
        model,
        processor,
        test_data,
        device,
        batch_size=args.batch_size,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        grader_model=args.grader_model,
        output_path=output_path,
    )
    print(f"\n{'=' * 50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
