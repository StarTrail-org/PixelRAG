#!/usr/bin/env python3
"""Local embedding: embed tile chunks using transformers on CPU or Apple MPS.

Works without CUDA — suitable for macOS (Apple Silicon), small-scale demos,
and testing.

Usage:
    # CPU (any platform)
    python -m pixelrag_embed.embed_cpu \
        --shard-dir ./tiles --output-dir ./embeddings

    # Apple Silicon GPU (macOS)
    python -m pixelrag_embed.embed_cpu \
        --shard-dir ./tiles --output-dir ./embeddings --device mps
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("embed_local")

Image.MAX_IMAGE_PIXELS = None

_RESIZE_FACTOR = 28
_MAX_CHUNK_WIDTH = 875


def _clamp_width(img: Image.Image, max_width: int = _MAX_CHUNK_WIDTH) -> Image.Image:
    """Resize so width <= max_width, preserving aspect ratio (28px alignment)."""
    w, h = img.size
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = max(round(w * scale / _RESIZE_FACTOR) * _RESIZE_FACTOR, _RESIZE_FACTOR)
    new_h = max(round(h * scale / _RESIZE_FACTOR) * _RESIZE_FACTOR, _RESIZE_FACTOR)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _resolve_device(device: str) -> str:
    """Resolve device string, auto-detecting MPS on macOS."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def scan_chunks(shard_dir: str) -> list[dict]:
    """Scan for chunk images in a shard directory.

    Looks for *.png.tiles/chunks.json files. Falls back to tiles.json if no chunks.
    """
    shard = Path(shard_dir)
    items = []

    for entry in sorted(shard.iterdir()):
        if not entry.is_dir():
            continue
        tile_dirs = []
        if entry.name.endswith(".png.tiles"):
            tile_dirs = [entry]
        else:
            tile_dirs = sorted(
                d
                for d in entry.iterdir()
                if d.is_dir() and d.name.endswith(".png.tiles")
            )

        for td in tile_dirs:
            dir_name = td.name
            article_id_str = dir_name.replace(".png.tiles", "")
            try:
                article_id = int(article_id_str)
            except ValueError:
                article_id = hash(article_id_str) % (2**31)

            chunks_json = td / "chunks.json"
            tiles_json = td / "tiles.json"

            if chunks_json.exists():
                with open(chunks_json) as f:
                    manifest = json.load(f)
                for chunk_info in manifest.get("chunks", []):
                    chunk_path = td / chunk_info["file"]
                    if chunk_path.exists():
                        items.append(
                            {
                                "path": str(chunk_path),
                                "article_id": article_id,
                                "tile_index": chunk_info.get("tile_index", 0),
                                "chunk_index": chunk_info.get("chunk_index", 0),
                                "y_offset": chunk_info.get("y_offset", 0),
                                "height": chunk_info.get("height", 1024),
                            }
                        )
            elif tiles_json.exists():
                with open(tiles_json) as f:
                    manifest = json.load(f)
                for i, tile_name in enumerate(manifest.get("tiles", [])):
                    tile_path = td / tile_name
                    if tile_path.exists():
                        items.append(
                            {
                                "path": str(tile_path),
                                "article_id": article_id,
                                "tile_index": i,
                                "chunk_index": 0,
                                "y_offset": 0,
                                "height": 0,
                            }
                        )

    return items


def embed_items(
    items: list[dict],
    model_name: str,
    device: str = "cpu",
    instruction: str = "",
) -> np.ndarray:
    """Embed image items using transformers on the given device."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = _resolve_device(device)
    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info("Loading model %s on %s (%s)...", model_name, device, dtype)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval()
    if device != "cpu":
        model = model.to(device)
    logger.info("Model loaded on %s", device)

    dim = model.config.text_config.hidden_size
    embeddings = np.zeros((len(items), dim), dtype=np.float16)

    prefix = f"Instruct: {instruction}\n" if instruction else ""

    for i, item in enumerate(tqdm(items, desc="Embedding")):
        img = Image.open(item["path"]).convert("RGB")
        img = _clamp_width(img)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prefix + "What is shown in this image?"},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        if device != "cpu":
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            seq_lens = inputs["attention_mask"].sum(dim=1)
            last_idx = seq_lens - 1
            pooled = last_hidden[0, last_idx[0]]
            pooled = pooled / pooled.norm()
            embeddings[i] = pooled.cpu().numpy().astype(np.float16)

        if (i + 1) % 10 == 0:
            logger.info("Embedded %d/%d", i + 1, len(items))

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Local embedding (CPU / MPS / CUDA)")
    parser.add_argument(
        "--shard-dir", required=True, help="Directory with *.png.tiles/ subdirs"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for .npz")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--instruction", default="", help="Instruction prefix for queries"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to embed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    items = scan_chunks(args.shard_dir)
    if not items:
        logger.error("No chunks found in %s", args.shard_dir)
        return

    if args.limit and len(items) > args.limit:
        logger.info("Found %d chunks, limiting to %d", len(items), args.limit)
        items = items[: args.limit]
    else:
        logger.info("Found %d chunks to embed", len(items))
    embeddings = embed_items(items, args.model, device=args.device, instruction=args.instruction)

    output_path = Path(args.output_dir) / "shard_000.npz"
    np.savez(
        output_path,
        embeddings=embeddings,
        article_ids=np.array([it["article_id"] for it in items], dtype=np.int64),
        tile_indices=np.array([it["tile_index"] for it in items], dtype=np.int32),
        chunk_indices=np.array([it["chunk_index"] for it in items], dtype=np.int32),
        y_offsets=np.array([it["y_offset"] for it in items], dtype=np.int32),
        tile_heights=np.array([it["height"] for it in items], dtype=np.int32),
    )

    logger.info("Saved %d embeddings to %s", len(items), output_path)


if __name__ == "__main__":
    main()
