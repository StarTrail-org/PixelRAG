#!/usr/bin/env python3
"""Core embedding pipeline: scan tiles from a shard, embed with vLLM, write .npz.

Usage (single shard):
    uv run embed_tiles.py \
        --shard-dir /opt/dlami/nvme/wiki-screenshot/output_coordinated/shard_042 \
        --output-dir ./output/shard_042 --gpu-ids 0

Multi-GPU:
    uv run embed_tiles.py \
        --shard-dir .../shard_042 --output-dir ./output/shard_042 --gpu-ids 0,1,2,3

Chunk embedding (default, recommended):
    Each 8192px tile is pre-split into 1024px strips (chunks) by the tiling pipeline.
    Chunks are stored alongside tiles in *.png.tiles/chunks.json.
    Embedding chunks instead of full tiles reduces the visual token count ~8x,
    significantly improving throughput.

    uv run embed_tiles.py \
        --shard-dir .../shard_042 \
        --output-dir ./output/shard_042 \
        --gpu-ids 0,1,2,3 \
        --mode chunks          # default, can be omitted
        --backend sglang \
        --batch-size 128

    Output npz arrays per chunk:
        embeddings      float16  [N, D]   — embedding vector
        article_ids     int64    [N]      — Wikipedia article ID
        tile_indices    int32    [N]      — which 8192px tile (0-based)
        chunk_indices   int32    [N]      — which 1024px strip within tile (0-based)
        y_offsets       int32    [N]      — Y position of chunk top edge in page (px)
        tile_heights    int32    [N]      — actual chunk height (last chunk may be <1024px)
        page_heights    int32    [N]      — full page height (px)
        viewport_widths int32    [N]      — page render width (px, capped at 875)
        image_hashes    S32      [N]      — MD5 of source PNG (used for dedup & patching)
        tile_paths      S512     [N]      — absolute path to chunk PNG file
        shard_id        int32    scalar   — shard number

    Lookup key: (article_id, tile_index, chunk_index) — lexsorted in output.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import queue
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

# Import multiprocessing at module level BEFORE atexit so mp's own
# _exit_function (which joins non-daemon children without timeout) is
# registered FIRST. Our _close_all_persistent_pools is registered last,
# so atexit LIFO order runs ours first — we force-kill stragglers before
# mp's handler can hang on them.
import multiprocessing  # noqa: F401
import multiprocessing.util  # noqa: F401
import atexit

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embed_tiles")
logger.setLevel(logging.INFO)

_PERSISTENT_POOLS: dict[tuple, "PersistentGpuWorkerPool"] = {}


def resolve_gpu_ids(gpu_ids_arg: str) -> list[int]:
    """Resolve GPU IDs from CLI arg.

    Supports:
    - "all" / "auto": use all visible GPUs
    - comma-separated IDs, e.g. "0,1,2,3"
    """
    value = gpu_ids_arg.strip().lower()
    if value in {"all", "auto"}:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            parts = [p.strip() for p in visible.split(",") if p.strip()]
            if parts:
                # Under CUDA_VISIBLE_DEVICES, local IDs are 0..N-1.
                return list(range(len(parts)))
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
            )
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if lines:
                return list(range(len(lines)))
        except Exception:
            logger.warning(
                "Failed to detect GPUs via nvidia-smi; falling back to GPU 0"
            )
        return [0]

    return [int(g.strip()) for g in gpu_ids_arg.split(",") if g.strip()]


_RESIZE_FACTOR = 28  # Qwen3-VL patch alignment
_MAX_CHUNK_WIDTH = 875  # web viewport width; wider images (e.g. PDF tiles) are resized


def _smart_resize_pil(img: "Image.Image", max_pixels: int) -> "Image.Image":
    """Resize image to fit within max_pixels, preserving aspect ratio.

    Dimensions are rounded to multiples of 28 (Qwen3-VL patch alignment).
    """
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(round(w * scale / _RESIZE_FACTOR) * _RESIZE_FACTOR, _RESIZE_FACTOR)
    new_h = max(round(h * scale / _RESIZE_FACTOR) * _RESIZE_FACTOR, _RESIZE_FACTOR)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _clamp_width_pil(img: "Image.Image", max_width: int = _MAX_CHUNK_WIDTH) -> "Image.Image":
    """Resize image so width <= max_width, preserving aspect ratio.

    Dimensions are rounded to multiples of 28 (Qwen3-VL patch alignment).
    Used for PDF tiles that render wider than the web viewport.
    """
    w, h = img.size
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = max(round(w * scale / _RESIZE_FACTOR) * _RESIZE_FACTOR, _RESIZE_FACTOR)
    new_h = max(round(h * scale / _RESIZE_FACTOR) * _RESIZE_FACTOR, _RESIZE_FACTOR)
    return img.resize((new_w, new_h), Image.LANCZOS)


def save_npz(path: str, compressed: bool, **arrays) -> None:
    """Save npz with optional compression."""
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Redirect filtering
# ---------------------------------------------------------------------------


def load_redirect_ids(path: str) -> set[int]:
    """Load redirect article IDs from a .redirects.json file.

    The file is a dict mapping article index (str) to target path (str).
    We only need the keys (indices of redirect articles).

    Args:
        path: Path to the .redirects.json file.

    Returns:
        Set of article indices that are client-side redirects.
    """
    with open(path, "r") as f:
        redirects = json.load(f)
    ids = {int(k) for k in redirects}
    logger.info("Loaded %d redirect IDs from %s", len(ids), path)
    return ids


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TileInfo(NamedTuple):
    """Metadata for a single tile image (or a chunk of one)."""

    article_id: int
    tile_index: int
    tile_path: str
    page_height: int
    viewport_width: int
    tile_height: int
    chunk_index: int = 0  # 0 = whole tile or first chunk


class ChunkInfo(NamedTuple):
    """Metadata for a single chunk image (1024px strip of a tile)."""

    article_id: int
    tile_index: int
    chunk_index: int
    chunk_path: str
    page_height: int
    viewport_width: int
    y_offset: int
    chunk_height: int


def _image_path(ti: "TileInfo | ChunkInfo") -> str:
    """Return the image file path regardless of info type."""
    return ti.tile_path if isinstance(ti, TileInfo) else ti.chunk_path


# ---------------------------------------------------------------------------
# Tile scanning
# ---------------------------------------------------------------------------


def scan_shard_tiles(
    shard_dir: str,
    skip_article_ids: set[int] | None = None,
) -> list[TileInfo]:
    """Walk a shard directory and collect all completed tiles.

    Looks for ``*.png.tiles/tiles.json`` files with ``complete: true``.
    Each tile PNG listed in tiles.json becomes one TileInfo entry.

    Args:
        shard_dir: Path to a shard directory (e.g. output_coordinated/shard_042).
        skip_article_ids: Article IDs to skip (already embedded).

    Returns:
        Sorted list of TileInfo (by article_id, then tile_index).
    """
    shard_path = Path(shard_dir)
    skip = skip_article_ids or set()
    tiles: list[TileInfo] = []

    # Shard directories contain sub-shard dirs (shard_00000, shard_00001, ...)
    # which contain article tile dirs (NNN.png.tiles/)
    tile_dirs: list[Path] = []
    for entry in sorted(shard_path.iterdir()):
        if entry.is_dir() and entry.name.startswith("shard_"):
            # Sub-shard directory
            for sub_entry in sorted(entry.iterdir()):
                if sub_entry.is_dir() and sub_entry.name.endswith(".png.tiles"):
                    tile_dirs.append(sub_entry)
        elif entry.is_dir() and entry.name.endswith(".png.tiles"):
            # Direct tile dir (flat shard layout)
            tile_dirs.append(entry)

    for tiles_dir in tile_dirs:
        tiles_json = tiles_dir / "tiles.json"
        if not tiles_json.exists():
            continue

        try:
            meta = json.loads(tiles_json.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", tiles_json, e)
            continue

        if not meta.get("complete", False):
            continue

        # Extract article_id from directory name: "3104240.png.tiles" -> 3104240
        dir_name = tiles_dir.name  # e.g. "3104240.png.tiles"
        try:
            article_id = int(dir_name.split(".")[0])
        except (ValueError, IndexError):
            logger.warning("Cannot parse article_id from %s", dir_name)
            continue

        if article_id in skip:
            continue

        page_height = meta.get("page_height", 0)
        viewport_width = meta.get("viewport_width", 875)
        tile_height = meta.get("tile_height", 8192)

        for idx, tile_name in enumerate(meta.get("tiles", [])):
            tile_path = tiles_dir / tile_name
            if tile_path.exists():
                tiles.append(
                    TileInfo(
                        article_id=article_id,
                        tile_index=idx,
                        tile_path=str(tile_path),
                        page_height=page_height,
                        viewport_width=viewport_width,
                        tile_height=tile_height,
                    )
                )

    tiles.sort(key=lambda t: (t.article_id, t.tile_index))
    logger.info(
        "Scanned %s: %d tiles from %d articles",
        shard_dir,
        len(tiles),
        len({t.article_id for t in tiles}),
    )
    return tiles


def scan_shard_chunks(
    shard_dir: str,
    skip_article_ids: set[int] | None = None,
) -> list[ChunkInfo]:
    """Walk a shard directory and collect all chunk images.

    Looks for ``*.png.tiles/chunks.json`` files. Each entry in
    ``meta["chunks"]`` becomes one ChunkInfo.

    Args:
        shard_dir: Path to a shard directory.
        skip_article_ids: Article IDs to skip (already embedded).

    Returns:
        Sorted list of ChunkInfo (by article_id, tile_index, chunk_index).
    """
    shard_path = Path(shard_dir)
    skip = skip_article_ids or set()
    chunks: list[ChunkInfo] = []

    tile_dirs: list[Path] = []
    for entry in sorted(shard_path.iterdir()):
        if entry.is_dir() and entry.name.startswith("shard_"):
            for sub_entry in sorted(entry.iterdir()):
                if sub_entry.is_dir() and sub_entry.name.endswith(".png.tiles"):
                    tile_dirs.append(sub_entry)
        elif entry.is_dir() and entry.name.endswith(".png.tiles"):
            tile_dirs.append(entry)

    for tiles_dir in tile_dirs:
        chunks_json = tiles_dir / "chunks.json"
        if not chunks_json.exists():
            continue

        try:
            meta = json.loads(chunks_json.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", chunks_json, e)
            continue

        dir_name = tiles_dir.name
        try:
            article_id = int(dir_name.split(".")[0])
        except (ValueError, IndexError):
            logger.warning("Cannot parse article_id from %s", dir_name)
            continue

        if article_id in skip:
            continue

        page_height = meta.get("page_height", 0)
        viewport_width = meta.get("viewport_width", 875)

        for chunk in meta.get("chunks", []):
            chunk_path = tiles_dir / chunk["file"]
            if chunk_path.exists():
                chunks.append(
                    ChunkInfo(
                        article_id=article_id,
                        tile_index=chunk["tile_index"],
                        chunk_index=chunk["chunk_index"],
                        chunk_path=str(chunk_path),
                        page_height=page_height,
                        viewport_width=viewport_width,
                        y_offset=chunk["y_offset"],
                        chunk_height=chunk["height"],
                    )
                )

    chunks.sort(key=lambda c: (c.article_id, c.tile_index, c.chunk_index))
    logger.info(
        "Scanned %s: %d chunks from %d articles",
        shard_dir,
        len(chunks),
        len({c.article_id for c in chunks}),
    )
    return chunks


# ---------------------------------------------------------------------------
# Tile reading and hashing (backend-agnostic)
# ---------------------------------------------------------------------------


def read_tiles_and_hash(
    tile_infos: list[TileInfo],
) -> list[tuple[TileInfo, str, "Image.Image"]]:
    """Read tile files, compute MD5 hashes, and load PIL images.

    Returns:
        List of (tile_info, md5_hex, pil_image) tuples for successfully loaded tiles.
    """
    results = []
    for ti in tile_infos:
        try:
            raw = Path(ti.tile_path).read_bytes()
            md5_hex = hashlib.md5(raw).hexdigest()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            results.append((ti, md5_hex, img))
        except Exception as e:
            logger.warning("Failed to load %s: %s", ti.tile_path, e)
    return results


def _read_hash_decode_one(
    ti: TileInfo,
) -> tuple[TileInfo, str, "Image.Image"] | None:
    """Read one tile, compute hash, and decode into RGB image."""
    try:
        raw = Path(ti.tile_path).read_bytes()
        md5_hex = hashlib.md5(raw).hexdigest()
        # Decode from in-memory bytes to avoid a second filesystem read.
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return (ti, md5_hex, img)
    except Exception as e:
        logger.warning("Failed to load %s: %s", ti.tile_path, e)
        return None


def read_tiles_and_hash_parallel(
    tile_infos: list[TileInfo],
    io_workers: int = 8,
) -> list[tuple[TileInfo, str, "Image.Image"]]:
    """Parallel tile read/hash/decode using a thread pool."""
    if io_workers <= 1 or len(tile_infos) <= 1:
        return read_tiles_and_hash(tile_infos)

    results: list[tuple[TileInfo, str, "Image.Image"]] = []
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for item in pool.map(_read_hash_decode_one, tile_infos, chunksize=32):
            if item is not None:
                results.append(item)
    return results


DEFAULT_INSTRUCTION = "Represent the user's input."

# Module-level instruction used by all embedding backends.
# Set via set_instruction() before starting workers to change from default.
_INSTRUCTION = DEFAULT_INSTRUCTION


def set_instruction(instruction: str) -> None:
    """Set the embedding instruction for all backends."""
    global _INSTRUCTION
    _INSTRUCTION = instruction
    logger.info("Embedding instruction set to: %r", instruction)


def _build_chat_prompt(tokenizer, instruction: str | None = None) -> str:
    """Build the embedding prompt using the model's chat template.

    Uses the official Qwen3-VL-Embedding prompt format:
        system: <instruction>
        user: [image]
    """
    instr = instruction if instruction is not None else _INSTRUCTION
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instr}]},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Backend: vLLM
# ---------------------------------------------------------------------------


def _init_vllm(model_path: str, gpu_id: int, enforce_eager: bool = False):
    """Initialize vLLM embedding engine on a single GPU."""
    from vllm import LLM, EngineArgs
    from vllm.config import PoolerConfig

    llm = LLM(
        **vars(
            EngineArgs(
                model=model_path,
                runner="pooling",
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=4096,
                enforce_eager=enforce_eager,
                pooler_config=PoolerConfig(pooling_type="LAST"),
            )
        )
    )
    return llm


def _get_tokenizer_vllm(engine):
    # vLLM API changed across versions:
    # - older: engine.llm_engine.tokenizer.tokenizer
    # - newer: engine.get_tokenizer()
    if hasattr(engine, "get_tokenizer"):
        return engine.get_tokenizer()
    return engine.llm_engine.tokenizer.tokenizer


def _embed_vllm(engine, prompt: str, images: list["Image.Image"]) -> list[np.ndarray]:
    """Run embedding on a batch of images via vLLM."""
    inputs = [{"prompt": prompt, "multi_modal_data": {"image": img}} for img in images]
    outputs = engine.embed(inputs)
    # L2 normalize (vLLM pooler does not normalize internally)
    embs = np.array([out.outputs.embedding for out in outputs], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-12)
    return [embs[i].astype(np.float16) for i in range(len(embs))]


# ---------------------------------------------------------------------------
# Backend: SGLang
# ---------------------------------------------------------------------------


def _init_sglang(model_path: str, gpu_id: int, enforce_eager: bool = False):
    """Initialize SGLang embedding engine on a single GPU.

    Key settings for vision embedding throughput on H100:
    - disable_cuda_graph: no decode phase in embedding → graphs waste memory
    - chunked_prefill_size=16384: large prefill chunks for batched vision tokens
    - max_prefill_tokens=65536: high ceiling so scheduler doesn't throttle
    - keep_mm_feature_on_device=True: avoid GPU→CPU→GPU copies of ViT features
    - mm_max_concurrent_calls=64: parallel image preprocessing
    - max_running_requests=512: pack more embedding requests
    - schedule_conservativeness=0.3: admit requests aggressively (KV freed instantly)
    - mem_fraction_static=0.80: safe with cuda graphs disabled
    - disable_radix_cache=True: embedding has no KV reuse
    """
    from sglang.srt.entrypoints.engine import Engine

    os.environ.setdefault("SGLANG_LOG_LEVEL", "error")
    return Engine(
        model_path=model_path,
        is_embedding=True,
        dtype="bfloat16",
        trust_remote_code=True,
        # Prefill tuning
        chunked_prefill_size=16384,
        max_prefill_tokens=65536,
        # Memory — no CUDA graphs frees several GB for activations
        mem_fraction_static=0.82,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        # Vision
        keep_mm_feature_on_device=True,
        mm_max_concurrent_calls=64,
        # Scheduling — aggressive for embedding (no decode, KV freed instantly)
        max_running_requests=512,
        schedule_conservativeness=0.3,
    )


def _get_tokenizer_sglang(engine):
    return engine.tokenizer_manager.tokenizer


def _embed_sglang(engine, prompt: str, images: list["Image.Image"]) -> list[np.ndarray]:
    """Run embedding on a batch via SGLang engine.encode().

    SGLang encode() takes prompts + image_data separately.
    Image data is passed as PIL Images (supported since sglang 0.5.x).
    """
    prompts = [prompt] * len(images)
    outputs = engine.encode(prompts, image_data=images)
    # L2 normalize (sglang encode() does not normalize internally)
    embs = np.array([out["embedding"] for out in outputs], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-12)
    return [embs[i].astype(np.float16) for i in range(len(embs))]


# ---------------------------------------------------------------------------
# Backend: direct_gpu (transformers + GPU-accelerated preprocessing)
# ---------------------------------------------------------------------------


def _init_direct_gpu(
    model_path: str,
    gpu_id: int,
    enforce_eager: bool = False,
    adapter_path: str | None = None,
):
    """Load model + processor for direct GPU inference.

    Uses Qwen3VLForConditionalGeneration (not AutoModel, which loads the
    base Qwen3VLModel with uninitialized language_model weights).
    GPU-accelerated image preprocessing (device='cuda' in processor) avoids
    the CPU preprocessing bottleneck: 0.2s vs 12s per batch of 64.

    If adapter_path is given, loads a PEFT LoRA adapter and merges it into
    the base weights (merge_and_unload) so inference runs at base-model speed.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        from peft import PeftModel, set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors
        import re as _re

        logger.info("GPU %d: loading LoRA adapter from %s", gpu_id, adapter_path)
        # The adapter was trained on Qwen3VLModel (BiQwen3), where attention
        # layers live at language_model.layers.*.  But Qwen3VLForConditional-
        # Generation wraps that inside self.model, so PEFT expects an extra
        # "model." prefix.  Remap keys before loading.
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        raw_sd = load_safetensors(adapter_weights_path)
        remapped_sd = {}
        for k, v in raw_sd.items():
            # base_model.model.language_model.* -> base_model.model.model.language_model.*
            new_k = _re.sub(
                r"^(base_model\.model\.)(language_model\.|visual\.)", r"\1model.\2", k
            )
            remapped_sd[new_k] = v
        n_remapped = sum(
            1 for ok, nk in zip(raw_sd.keys(), remapped_sd.keys()) if ok != nk
        )
        logger.info(
            "GPU %d: remapped %d/%d adapter keys (BiQwen3 -> ConditionalGeneration)",
            gpu_id,
            n_remapped,
            len(raw_sd),
        )

        model = PeftModel.from_pretrained(model, adapter_path)
        set_peft_model_state_dict(model, remapped_sd)
        model = model.merge_and_unload()
        logger.info("GPU %d: LoRA merged into base weights", gpu_id)

    model = model.cuda().eval()

    # WORKAROUND: torch 2.9.x Conv3d bf16 bug (pytorch/pytorch#166122) —
    # cuDNN 9.8–9.14 disabled for 3D conv, vol2col fallback is 16,400x slower for bf16.
    # Fixed in cuDNN >= 9.15. Only apply fp32 workaround if needed.
    cudnn_ver = torch.backends.cudnn.version()
    if cudnn_ver < 91500:
        _pe = model.model.visual.patch_embed

        def _fp32_patch_embed(hidden_states, _pe=_pe):
            conv = _pe.proj
            x = hidden_states.view(
                -1,
                _pe.in_channels,
                _pe.temporal_patch_size,
                _pe.patch_size,
                _pe.patch_size,
            )
            old_w, old_b = conv.weight.data, conv.bias.data
            conv.weight.data = old_w.float()
            conv.bias.data = old_b.float()
            out = conv(x.float()).view(-1, _pe.embed_dim)
            conv.weight.data = old_w
            conv.bias.data = old_b
            return out.to(torch.bfloat16)

        _pe.forward = _fp32_patch_embed
        logger.info(f"Applied fp32 Conv3d workaround (cuDNN {cudnn_ver} < 91500)")
    else:
        logger.info(f"Native bf16 Conv3d (cuDNN {cudnn_ver} >= 91500, bug fixed)")

    return (model, processor)


def _get_tokenizer_direct_gpu(engine):
    _, processor = engine
    return processor.tokenizer


def _embed_direct_gpu(
    engine, prompt: str, images: list["Image.Image"]
) -> list[np.ndarray]:
    """Embed images using direct model forward with GPU preprocessing.

    The prompt arg is the chat template text (same format as sglang).
    We pass device='cuda' to the processor so all image tensor ops
    (resize, normalize, stack, cat) run on GPU instead of CPU.
    """
    import torch

    model, processor = engine

    # Build per-image message format for the processor
    messages_batch = [
        [
            {"role": "system", "content": [{"type": "text", "text": _INSTRUCTION}]},
            {"role": "user", "content": [{"type": "image", "image": img}]},
        ]
        for img in images
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    # GPU-accelerated preprocessing: device="cuda" moves resize/normalize/stack to GPU
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True, device="cuda"
    )
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model(**inputs)

    # Last-token pooling — use last_hidden_state (post-RMSNorm)
    last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"]  # (batch, seq_len)
    # Find the last non-padding token for each sequence
    last_token_indices = attention_mask.sum(dim=1) - 1  # (batch,)
    pooled = last_hidden[
        torch.arange(last_hidden.size(0), device=last_hidden.device), last_token_indices
    ]
    # L2 normalize
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

    return [emb.cpu().float().numpy().astype(np.float16) for emb in pooled]


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

BACKENDS = {
    "vllm": (_init_vllm, _get_tokenizer_vllm, _embed_vllm),
    "sglang": (_init_sglang, _get_tokenizer_sglang, _embed_sglang),
    "direct_gpu": (_init_direct_gpu, _get_tokenizer_direct_gpu, _embed_direct_gpu),
}


# ---------------------------------------------------------------------------
# Single-GPU worker (runs in a subprocess via torch.multiprocessing)
# ---------------------------------------------------------------------------


def gpu_worker(
    gpu_id: int,
    tile_infos: list[TileInfo],
    model_path: str,
    batch_size: int,
    result_dir: str,
    backend: str = "vllm",
    io_workers: int = 8,
    compress_npz: bool = False,
    max_pixels: int | None = None,
    chunk_height: int | None = None,
    enforce_eager: bool = False,
    init_barrier=None,
    adapter_path: str | None = None,
) -> str:
    """Load model on one GPU, embed all assigned tiles, write partial .npz.

    This function runs in a child process with CUDA_VISIBLE_DEVICES set.

    Args:
        gpu_id: Physical GPU ID.
        tile_infos: Tiles assigned to this GPU.
        model_path: HuggingFace model name or local path.
        batch_size: Tiles per embed call.
        result_dir: Directory to write partial result file.
        backend: "vllm" or "sglang".
        init_barrier: If provided, wait here after model load so all GPUs
            finish CUDA graph capture before any starts embedding.  Avoids
            concurrent-capture stalls on some GPUs (observed on GPU 6/vLLM).

    Returns:
        Path to the partial .npz file written by this worker.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("SGLANG_LOG_LEVEL", "error")

    init_fn, get_tok_fn, embed_fn = BACKENDS[backend]

    logger.info(
        "GPU %d: loading model %s via %s (%d tiles)",
        gpu_id,
        model_path,
        backend,
        len(tile_infos),
    )
    t0 = time.time()

    # Pass adapter_path only for direct_gpu (other backends don't support it)
    if adapter_path and backend == "direct_gpu":
        engine = init_fn(
            model_path, gpu_id, enforce_eager=enforce_eager, adapter_path=adapter_path
        )
    else:
        engine = init_fn(model_path, gpu_id, enforce_eager=enforce_eager)

    dt = time.time() - t0
    logger.info("GPU %d: model loaded in %.1fs", gpu_id, dt)

    # Wait for all GPUs to finish init (CUDA graph capture) before embedding.
    if init_barrier is not None:
        try:
            init_barrier.wait(timeout=120)
            logger.info("GPU %d: all workers ready, starting embed", gpu_id)
        except threading.BrokenBarrierError:
            logger.error(
                "GPU %d: barrier broken (another GPU failed to init), continuing anyway",
                gpu_id,
            )

    def _cleanup_engine() -> None:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as e:
                logger.warning("GPU %d: engine shutdown failed: %s", gpu_id, e)

    # Build prompt via chat template (official Qwen3-VL-Embedding format)
    tokenizer = get_tok_fn(engine)
    prompt = _build_chat_prompt(tokenizer)
    logger.info("GPU %d: prompt template: %s", gpu_id, repr(prompt[:120]))

    partial_path = _embed_tile_infos_with_engine(
        engine=engine,
        gpu_id=gpu_id,
        tile_infos=tile_infos,
        batch_size=batch_size,
        result_dir=result_dir,
        embed_fn=embed_fn,
        prompt=prompt,
        io_workers=io_workers,
        compress_npz=compress_npz,
        max_pixels=max_pixels,
        chunk_height=chunk_height,
    )
    _cleanup_engine()
    return partial_path


_MIN_CHUNK_HEIGHT = 28  # One Qwen3-VL patch; smaller chunks cause processor errors


def _chunk_image(img: "Image.Image", chunk_height: int) -> list["Image.Image"]:
    """Split a tall image into horizontal strips of chunk_height pixels.

    The last chunk is merged into the previous one if it would be shorter
    than _MIN_CHUNK_HEIGHT pixels.
    """
    w, h = img.size
    if h <= chunk_height:
        return [img]
    chunks = []
    for y in range(0, h, chunk_height):
        y_end = min(y + chunk_height, h)
        # Merge tiny remainder into previous chunk
        if y_end - y < _MIN_CHUNK_HEIGHT and chunks:
            prev = chunks[-1]
            merged = Image.new("RGB", (w, prev.size[1] + y_end - y))
            merged.paste(prev, (0, 0))
            merged.paste(img.crop((0, y, w, y_end)), (0, prev.size[1]))
            chunks[-1] = merged
        else:
            chunks.append(img.crop((0, y, w, y_end)))
    return chunks


def _embed_tile_infos_with_engine(
    engine,
    gpu_id: int,
    tile_infos: list[TileInfo],
    batch_size: int,
    result_dir: str,
    embed_fn,
    prompt: str,
    io_workers: int = 8,
    compress_npz: bool = False,
    max_pixels: int | None = None,
    chunk_height: int | None = None,
    task_id: str | None = None,
) -> str:
    """Embed tile infos with a pre-initialized engine function + prompt.

    Uses a pipeline: background threads read files, hash, and decode only
    unique tiles while the main thread feeds decoded batches to the GPU.
    Duplicate tiles are hashed but never decoded.
    If chunk_height is set, tall tiles are split into chunks of that height.
    If max_pixels is set, images are resized in the I/O threads (overlaps GPU).
    """
    tile_hashes: list[tuple[TileInfo, str]] = []  # filled by producer
    unique_queue: queue.Queue = queue.Queue(maxsize=batch_size * 2)
    producer_error: list[Exception | None] = [None]

    resized_wide = [0]  # mutable counter for threads

    def _io_producer() -> None:
        """Read tiles, chunk tall images, hash chunks, skip decode for duplicates."""
        seen: set[str] = set()
        lock = threading.Lock()
        results_lock = threading.Lock()

        def _read_chunk_hash(ti):
            img_path = _image_path(ti)
            try:
                raw = Path(img_path).read_bytes()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                logger.warning("Skipping unreadable image %s: %s", img_path, e)
                return []

            # Chunk tall images into strips (skip if already pre-chunked)
            if chunk_height is not None and not isinstance(ti, ChunkInfo):
                chunks = _chunk_image(img, chunk_height)
            else:
                chunks = [img]

            out = []
            for ci, chunk_img in enumerate(chunks):
                if max_pixels is not None:
                    chunk_img = _smart_resize_pil(chunk_img, max_pixels)
                cw, ch = chunk_img.size
                # Skip chunks with extreme aspect ratio (sglang rejects >200)
                if ch < 5 or (cw / max(ch, 1)) > 150:
                    logger.warning(
                        "Skipping bad chunk %s ci=%d size=%dx%d (aspect ratio %.1f)",
                        img_path,
                        ci,
                        cw,
                        ch,
                        cw / max(ch, 1),
                    )
                    continue
                if cw > _MAX_CHUNK_WIDTH:
                    chunk_img = _clamp_width_pil(chunk_img, _MAX_CHUNK_WIDTH)
                    cw, ch = chunk_img.size
                    resized_wide[0] += 1
                    logger.debug(
                        "Resized wide chunk %s ci=%d to %dx%d",
                        img_path, ci, cw, ch,
                    )
                # Dedup key: use file path for pre-chunked files (avoids
                # expensive tobytes + MD5 on ~2.7MB raw pixels per chunk)
                if isinstance(ti, ChunkInfo):
                    h = img_path  # unique on disk, no hash needed
                else:
                    chunk_bytes = chunk_img.tobytes()
                    h = hashlib.md5(chunk_bytes).hexdigest()
                if len(chunks) > 1:
                    if isinstance(ti, ChunkInfo):
                        chunk_ti = ti._replace(
                            chunk_index=ci,
                            chunk_height=chunk_img.size[1],
                        )
                    else:
                        chunk_ti = ti._replace(
                            chunk_index=ci,
                            tile_height=chunk_img.size[1],
                        )
                else:
                    chunk_ti = ti
                with lock:
                    is_new = h not in seen
                    if is_new:
                        seen.add(h)
                out.append((chunk_ti, h, chunk_img if is_new else None))
            return out

        try:
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                for items in pool.map(_read_chunk_hash, tile_infos):
                    for chunk_ti, h, img in items:
                        with results_lock:
                            tile_hashes.append((chunk_ti, h))
                        if img is not None:
                            unique_queue.put((h, img))
        except Exception as exc:
            producer_error[0] = exc
        finally:
            unique_queue.put(None)  # sentinel

    t0 = time.time()
    producer = threading.Thread(target=_io_producer, daemon=True)
    producer.start()

    # Consume unique decoded tiles and embed in batches (overlaps with I/O)
    hash_to_embedding: dict[str, np.ndarray] = {}
    batch_h: list[str] = []
    batch_imgs: list = []
    unique_total = 0
    embedded = 0
    embed_time_total = 0.0
    queue_wait_total = 0.0
    n_batches = 0
    first_item_time = None
    consecutive_failures = 0
    _MAX_CONSECUTIVE_FAILURES = (
        3  # after 3 consecutive CUDA failures, abort this work item
    )
    gpu_dead = False

    while True:
        tq0 = time.time()
        item = unique_queue.get()
        queue_wait_total += time.time() - tq0
        if item is None:
            break
        if first_item_time is None:
            first_item_time = time.time() - t0
        h, img = item
        batch_h.append(h)
        batch_imgs.append(img)
        unique_total += 1

        if len(batch_h) >= batch_size:
            if gpu_dead:
                # Drain queue without embedding
                batch_h, batch_imgs = [], []
                continue
            try:
                te0 = time.time()
                embs = embed_fn(engine, prompt, batch_imgs)
                embed_time_total += time.time() - te0
                n_batches += 1
                consecutive_failures = 0
                for hh, emb in zip(batch_h, embs):
                    hash_to_embedding[hh] = emb
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    "GPU %d: embed failed at offset %d (%d/%d consecutive): %s",
                    gpu_id,
                    embedded,
                    consecutive_failures,
                    _MAX_CONSECUTIVE_FAILURES,
                    e,
                )
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "GPU %d: %d consecutive failures — marking GPU as dead",
                        gpu_id,
                        consecutive_failures,
                    )
                    gpu_dead = True
            embedded += len(batch_h)
            batch_h, batch_imgs = [], []

    # Final partial batch
    if batch_h and not gpu_dead:
        try:
            te0 = time.time()
            embs = embed_fn(engine, prompt, batch_imgs)
            embed_time_total += time.time() - te0
            n_batches += 1
            for hh, emb in zip(batch_h, embs):
                hash_to_embedding[hh] = emb
        except Exception as e:
            logger.error("GPU %d: embed failed on final batch: %s", gpu_id, e)
        embedded += len(batch_h)

    producer.join()
    pipeline_s = time.time() - t0

    if producer_error[0]:
        logger.error("GPU %d: I/O producer failed: %s", gpu_id, producer_error[0])

    deduped = len(tile_hashes) - unique_total
    100.0 * deduped / len(tile_hashes) if tile_hashes else 0.0
    n_resized = resized_wide[0]
    logger.info(
        "GPU %d: %d images, %d unique (%d deduped, %d resized>%d), embedded %d, pipeline %.2fs (%.1f chunks/s)"
        " | embed=%.2fs (%d batches) queue_wait=%.2fs first_item=%.3fs",
        gpu_id,
        len(tile_hashes),
        unique_total,
        deduped,
        n_resized,
        _MAX_CHUNK_WIDTH,
        embedded,
        pipeline_s,
        embedded / pipeline_s if pipeline_s > 0 else 0,
        embed_time_total,
        n_batches,
        queue_wait_total,
        first_item_time or 0,
    )

    if not hash_to_embedding:
        logger.warning("GPU %d: no embeddings produced", gpu_id)
        if gpu_dead:
            raise RuntimeError(
                f"GPU {gpu_id}: CUDA dead after {_MAX_CONSECUTIVE_FAILURES} consecutive failures"
            )
        return ""

    # Detect chunk mode from the type of items in tile_hashes
    is_chunk_mode = tile_hashes and isinstance(tile_hashes[0][0], ChunkInfo)

    # Phase 3: expand back to all tiles/chunks, reusing embeddings for duplicate hashes
    all_embeddings = []
    all_article_ids = []
    all_tile_indices = []
    all_chunk_indices = []
    all_page_heights = []
    all_viewport_widths = []
    all_tile_heights = []
    all_image_hashes = []
    all_tile_paths = []
    all_y_offsets = [] if is_chunk_mode else None

    for ti, h in tile_hashes:
        if h not in hash_to_embedding:
            continue  # embedding failed for this hash
        all_embeddings.append(hash_to_embedding[h])
        all_article_ids.append(ti.article_id)
        all_tile_indices.append(ti.tile_index)
        all_page_heights.append(ti.page_height)
        all_viewport_widths.append(ti.viewport_width)
        all_image_hashes.append(h)
        if is_chunk_mode:
            all_chunk_indices.append(ti.chunk_index)
            all_tile_heights.append(ti.chunk_height)
            all_tile_paths.append(ti.chunk_path)
            all_y_offsets.append(ti.y_offset)
        else:
            all_chunk_indices.append(ti.chunk_index)
            all_tile_heights.append(ti.tile_height)
            all_tile_paths.append(ti.tile_path)

    if not all_embeddings:
        logger.warning("GPU %d: no embeddings after expansion", gpu_id)
        return ""

    suffix = f"_{task_id}" if task_id else ""
    partial_path = os.path.join(result_dir, f"partial_gpu{gpu_id}{suffix}.npz")
    t_write0 = time.time()
    extra_arrays = {}
    if is_chunk_mode:
        extra_arrays["y_offsets"] = np.array(all_y_offsets, dtype=np.int32)
    save_npz(
        partial_path,
        compressed=compress_npz,
        embeddings=np.stack(all_embeddings),
        article_ids=np.array(all_article_ids, dtype=np.int64),
        tile_indices=np.array(all_tile_indices, dtype=np.int32),
        chunk_indices=np.array(all_chunk_indices, dtype=np.int32),
        page_heights=np.array(all_page_heights, dtype=np.int32),
        viewport_widths=np.array(all_viewport_widths, dtype=np.int32),
        tile_heights=np.array(all_tile_heights, dtype=np.int32),
        image_hashes=np.array(all_image_hashes, dtype="S32"),
        tile_paths=np.array(all_tile_paths, dtype="S512"),
        **extra_arrays,
    )
    write_s = time.time() - t_write0
    logger.info(
        "GPU %d: wrote %d embeddings (%d unique) to %s [pipeline %.2fs, write %.2fs]",
        gpu_id,
        len(all_embeddings),
        len(hash_to_embedding),
        partial_path,
        pipeline_s,
        write_s,
    )
    return partial_path


def _gpu_worker_entry(
    result_queue,
    gpu_id: int,
    tile_infos: list[TileInfo],
    model_path: str,
    batch_size: int,
    result_dir: str,
    backend: str,
    io_workers: int,
    compress_npz: bool,
    max_pixels: int | None = None,
    chunk_height: int | None = None,
    enforce_eager: bool = False,
    init_barrier=None,
    adapter_path: str | None = None,
) -> None:
    """Run gpu_worker in a subprocess and report result via queue."""
    try:
        partial_path = gpu_worker(
            gpu_id=gpu_id,
            tile_infos=tile_infos,
            model_path=model_path,
            batch_size=batch_size,
            result_dir=result_dir,
            backend=backend,
            io_workers=io_workers,
            compress_npz=compress_npz,
            max_pixels=max_pixels,
            chunk_height=chunk_height,
            enforce_eager=enforce_eager,
            init_barrier=init_barrier,
            adapter_path=adapter_path,
        )
        result_queue.put(
            {
                "gpu_id": gpu_id,
                "partial_path": partial_path,
                "error": "",
            }
        )
    except Exception:
        result_queue.put(
            {
                "gpu_id": gpu_id,
                "partial_path": "",
                "error": traceback.format_exc(),
            }
        )


def _gpu_worker_persistent_entry(
    work_queue,
    result_queue,
    gpu_id: int,
    model_path: str,
    backend: str,
    io_workers: int,
    compress_npz: bool,
    max_pixels: int | None = None,
    chunk_height: int | None = None,
    enforce_eager: bool = False,
    init_barrier=None,
    adapter_path: str | None = None,
) -> None:
    """Persistent GPU worker: load model once, pull work dynamically from shared queue.

    Protocol:
        - Work items: dict with "task_id", "tile_infos", "batch_size", "result_dir"
        - Round sentinel: dict with "task_id"=None, "round_id" — worker sends round_done and continues
        - Shutdown: None — worker exits
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("SGLANG_LOG_LEVEL", "error")
    init_fn, get_tok_fn, embed_fn = BACKENDS[backend]

    engine = None
    try:
        if adapter_path and backend == "direct_gpu":
            engine = init_fn(
                model_path,
                gpu_id,
                enforce_eager=enforce_eager,
                adapter_path=adapter_path,
            )
        else:
            engine = init_fn(model_path, gpu_id, enforce_eager=enforce_eager)
        if init_barrier is not None:
            try:
                init_barrier.wait(timeout=120)
                logger.info("GPU %d: all persistent workers ready", gpu_id)
            except threading.BrokenBarrierError:
                logger.error("GPU %d: barrier broken, continuing anyway", gpu_id)
        tokenizer = get_tok_fn(engine)
        prompt = _build_chat_prompt(tokenizer)

        while True:
            task = work_queue.get()
            if task is None:
                # Shutdown signal
                break
            if task.get("task_id") is None:
                # Round-end sentinel — signal done for this round, keep looping
                result_queue.put(
                    {
                        "gpu_id": gpu_id,
                        "round_done": True,
                        "round_id": task.get("round_id"),
                    }
                )
                continue
            task_id = task["task_id"]
            try:
                partial_path = _embed_tile_infos_with_engine(
                    engine=engine,
                    gpu_id=gpu_id,
                    tile_infos=task["tile_infos"],
                    batch_size=task["batch_size"],
                    result_dir=task["result_dir"],
                    embed_fn=embed_fn,
                    prompt=prompt,
                    io_workers=io_workers,
                    compress_npz=compress_npz,
                    max_pixels=max_pixels,
                    chunk_height=chunk_height,
                    task_id=task_id,
                )
                result_queue.put(
                    {
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "partial_path": partial_path,
                        "error": "",
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "partial_path": "",
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        result_queue.put(
            {
                "task_id": "__init__",
                "gpu_id": gpu_id,
                "partial_path": "",
                "error": traceback.format_exc(),
            }
        )
    finally:
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass


_WORK_CHUNK_SIZE = 500  # tiles per dynamic work item (good balance: ~4 batches of 128)


class PersistentGpuWorkerPool:
    """Pool of persistent one-GPU workers with dynamic work distribution.

    Workers share a single work queue. Fast GPUs pull more work items,
    eliminating idle time from static pre-splitting.
    """

    def __init__(
        self,
        gpu_ids: list[int],
        model_path: str,
        backend: str,
        io_workers: int,
        compress_npz: bool,
        max_pixels: int | None = None,
        chunk_height: int | None = None,
        enforce_eager: bool = False,
        adapter_path: str | None = None,
    ) -> None:
        import torch.multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.work_queue = self.ctx.Queue()  # shared across all workers
        self.result_queue = self.ctx.Queue()
        self.gpu_ids = list(gpu_ids)
        self.workers: dict[int, "mp.Process"] = {}
        # Store init params for respawning dead workers
        self._model_path = model_path
        self._backend = backend
        self._io_workers = io_workers
        self._compress_npz = compress_npz
        self._max_pixels = max_pixels
        self._chunk_height = chunk_height
        self._enforce_eager = enforce_eager
        self._adapter_path = adapter_path
        # Store barrier as instance var to prevent GC before children unpickle
        self._init_barrier = (
            self.ctx.Barrier(len(gpu_ids)) if len(gpu_ids) > 1 else None
        )
        self._gpu_death_count: dict[int, int] = {}  # gid -> number of times worker died
        self._excluded_gpus: set[int] = set()  # permanently excluded GPUs

        for gid in gpu_ids:
            p = self.ctx.Process(
                target=_gpu_worker_persistent_entry,
                args=(
                    self.work_queue,
                    self.result_queue,
                    gid,
                    model_path,
                    backend,
                    io_workers,
                    compress_npz,
                    max_pixels,
                    chunk_height,
                    enforce_eager,
                    self._init_barrier,
                    adapter_path,
                ),
                daemon=False,
            )
            p.start()
            self.workers[gid] = p

    def run(
        self,
        tile_infos: list[TileInfo],
        batch_size: int,
        result_dir: str,
    ) -> list[str]:
        """Distribute tile_infos dynamically across workers and collect results.

        Splits tile_infos into small work chunks and puts them into the shared
        queue. Workers pull chunks as they finish, so fast GPUs process more.
        """
        round_id = f"{time.time_ns()}"
        t_round_start = time.time()

        # Respawn dead workers from previous rounds (skip permanently excluded GPUs)
        dead_before = [
            gid
            for gid, p in self.workers.items()
            if not p.is_alive() and gid not in self._excluded_gpus
        ]
        for gid in dead_before:
            self._gpu_death_count[gid] = self._gpu_death_count.get(gid, 0) + 1
            if self._gpu_death_count[gid] > 2:
                logger.error(
                    "GPU %d: died %d times — permanently excluding from pool",
                    gid,
                    self._gpu_death_count[gid],
                )
                self._excluded_gpus.add(gid)
                self.workers[gid].close()
                del self.workers[gid]
                continue
            logger.warning(
                "GPU %d: worker died (exitcode=%s), respawning (death #%d)",
                gid,
                self.workers[gid].exitcode,
                self._gpu_death_count[gid],
            )
            self.workers[gid].close()
            p = self.ctx.Process(
                target=_gpu_worker_persistent_entry,
                args=(
                    self.work_queue,
                    self.result_queue,
                    gid,
                    self._model_path,
                    self._backend,
                    self._io_workers,
                    self._compress_npz,
                    self._max_pixels,
                    self._chunk_height,
                    self._enforce_eager,
                    None,
                    self._adapter_path,
                ),  # no barrier for respawned workers
                daemon=False,
            )
            p.start()
            self.workers[gid] = p
            logger.info("GPU %d: respawned worker (pid=%d)", gid, p.pid)
        if dead_before:
            # Give respawned workers time to load model
            logger.info(
                "Waiting 30s for %d respawned workers to load model...",
                len(dead_before),
            )
            time.sleep(30)
        n_workers = len(self.workers)
        if n_workers == 0:
            raise RuntimeError("All persistent workers are dead, cannot run")

        # Split into work chunks
        work_items = []
        for i in range(0, len(tile_infos), _WORK_CHUNK_SIZE):
            chunk = tile_infos[i : i + _WORK_CHUNK_SIZE]
            work_items.append(
                {
                    "task_id": f"{round_id}_{len(work_items)}",
                    "round_id": round_id,
                    "tile_infos": chunk,
                    "batch_size": batch_size,
                    "result_dir": result_dir,
                }
            )

        logger.info(
            "Dynamic distribution: %d tiles -> %d work items for %d GPUs",
            len(tile_infos),
            len(work_items),
            n_workers,
        )

        # Enqueue all work items (no sentinels — we count results instead)
        for item in work_items:
            self.work_queue.put(item)
        n_work_items = len(work_items)

        # Collect results until we've received one result per work item
        partial_paths: list[str] = []
        errors: list[str] = []
        results_received = 0
        gpu_work_counts: dict[int, int] = {}
        gpu_consecutive_errors: dict[int, int] = {}  # track per-GPU error streaks

        dead_reported: set[int] = set()
        stall_count = 0  # consecutive timeouts with no new results

        while results_received < n_work_items:
            try:
                msg = self.result_queue.get(timeout=10)
                stall_count = 0  # got something, reset
            except queue.Empty:
                stall_count += 1
                # Check for dead workers
                for gid, p in self.workers.items():
                    if gid not in dead_reported and not p.is_alive():
                        dead_reported.add(gid)
                        errors.append(f"GPU {gid}: worker died (exitcode={p.exitcode})")
                        logger.error(
                            "GPU %d: worker died (exitcode=%s), %d/%d results so far",
                            gid,
                            p.exitcode,
                            results_received,
                            n_work_items,
                        )
                alive = [gid for gid, p in self.workers.items() if p.is_alive()]
                if not alive:
                    remaining = n_work_items - results_received
                    errors.append(
                        f"All workers dead, {remaining} work items unprocessed"
                    )
                    break
                # If a worker died and we've stalled for 60s, remaining items
                # were likely in that worker's pipeline — stop waiting
                if dead_reported and stall_count >= 6:
                    remaining = n_work_items - results_received
                    logger.warning(
                        "Stalled 60s after worker death, giving up on %d remaining items",
                        remaining,
                    )
                    break
                continue

            if msg.get("round_done"):
                # Stale sentinel from a previous round — ignore
                continue

            # Normal work result
            results_received += 1
            gid = msg.get("gpu_id", -1)
            gpu_work_counts[gid] = gpu_work_counts.get(gid, 0) + 1

            if msg.get("error"):
                gpu_consecutive_errors[gid] = gpu_consecutive_errors.get(gid, 0) + 1
                if gpu_consecutive_errors[gid] >= 3 and gid not in dead_reported:
                    dead_reported.add(gid)
                    logger.error(
                        "GPU %d: %d consecutive task errors — treating as dead, "
                        "killing worker to prevent it from stealing work",
                        gid,
                        gpu_consecutive_errors[gid],
                    )
                    # Kill the worker process so it stops pulling from the shared queue
                    p = self.workers.get(gid)
                    if p and p.is_alive():
                        p.kill()
                errors.append(
                    f"GPU {gid} task {msg.get('task_id')} failed:\n{msg['error']}"
                )
                continue
            # Successful result — reset error streak for this GPU
            gpu_consecutive_errors[gid] = 0
            pp = msg.get("partial_path", "")
            if pp:
                partial_paths.append(pp)

        # Log per-GPU work distribution and round summary
        round_elapsed = time.time() - t_round_start
        total_chunks = len(tile_infos)
        throughput = total_chunks / round_elapsed if round_elapsed > 0 else 0
        per_gpu_tp = throughput / n_workers if n_workers > 0 else 0
        gpu_summary = ", ".join(
            f"GPU {gid}: {gpu_work_counts.get(gid, 0)}" for gid in sorted(self.workers)
        )
        logger.info(
            "Round done: %d chunks in %.1fs = %.1f chunks/s (%.1f/GPU) | %s",
            total_chunks,
            round_elapsed,
            throughput,
            per_gpu_tp,
            gpu_summary,
        )

        if errors:
            logger.error("Dynamic worker errors:\n%s", "\n".join(errors))
            if not partial_paths:
                raise RuntimeError("All dynamic workers failed:\n" + "\n".join(errors))
            # If a GPU died and stole work, the shard is incomplete — raise so caller retries
            if dead_reported:
                raise RuntimeError(
                    f"GPU(s) {dead_reported} died during round — shard is incomplete "
                    f"({len(partial_paths)} partial results from {n_work_items} work items). "
                    f"Caller should retry without dead GPU(s)."
                )
            logger.warning(
                "%d errors, continuing with %d partial results",
                len(errors),
                len(partial_paths),
            )

        return partial_paths

    def close(self) -> None:
        # Send shutdown sentinels. Workers read None from work_queue and exit.
        for _ in self.workers:
            try:
                self.work_queue.put(None)
            except Exception:
                pass
        # Soft-join: give each worker a short window to exit cleanly.
        for _gid, p in self.workers.items():
            p.join(timeout=10)
        # Force-kill any stragglers so interpreter shutdown isn't blocked on
        # non-daemon children. CUDA context teardown in child can hang
        # indefinitely; SIGTERM/SIGKILL is safe here because results already
        # landed in result_queue and partial_path files before close() runs.
        for gid, p in list(self.workers.items()):
            if p.is_alive():
                logger.warning(
                    "GPU %d worker pid=%s didn't exit on sentinel, SIGTERM", gid, p.pid
                )
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                logger.error(
                    "GPU %d worker pid=%s ignored SIGTERM, SIGKILL", gid, p.pid
                )
                p.kill()
                p.join(timeout=5)
        # Drop queue background feeder threads so they don't block atexit.
        for q in (self.work_queue, self.result_queue):
            try:
                q.close()
                q.cancel_join_thread()
            except Exception:
                pass


def _close_all_persistent_pools() -> None:
    for _k, pool in list(_PERSISTENT_POOLS.items()):
        try:
            pool.close()
        except Exception:
            pass
    _PERSISTENT_POOLS.clear()


atexit.register(_close_all_persistent_pools)


def run_gpu_workers_parallel(
    per_gpu: dict[int, list[TileInfo]] | None,
    gpu_ids: list[int],
    model_path: str,
    batch_size: int,
    result_dir: str,
    backend: str,
    io_workers: int,
    compress_npz: bool,
    reuse_workers: bool = False,
    max_pixels: int | None = None,
    chunk_height: int | None = None,
    enforce_eager: bool = False,
    tile_infos: list[TileInfo] | None = None,
    adapter_path: str | None = None,
) -> list[str]:
    """Launch one non-daemonic process per GPU and collect partial .npz paths.

    When reuse_workers=True, uses dynamic work distribution via a shared queue.
    Pass tile_infos directly (per_gpu is ignored). Workers pull work chunks
    dynamically, so fast GPUs process more.

    When reuse_workers=False, uses static per_gpu assignment with one-shot processes.
    """
    if reuse_workers:
        # Dynamic distribution: pass all tiles, workers pull from shared queue
        all_tiles = tile_infos if tile_infos is not None else []
        if not all_tiles and per_gpu:
            # Fallback: flatten per_gpu into single list
            for tiles in per_gpu.values():
                all_tiles.extend(tiles)

        key = (
            tuple(gpu_ids),
            model_path,
            backend,
            io_workers,
            bool(compress_npz),
            max_pixels,
            chunk_height,
            enforce_eager,
            adapter_path,
        )
        pool = _PERSISTENT_POOLS.get(key)
        if pool is None:
            pool = PersistentGpuWorkerPool(
                gpu_ids=gpu_ids,
                model_path=model_path,
                backend=backend,
                io_workers=io_workers,
                compress_npz=compress_npz,
                max_pixels=max_pixels,
                chunk_height=chunk_height,
                enforce_eager=enforce_eager,
                adapter_path=adapter_path,
            )
            _PERSISTENT_POOLS[key] = pool
        return pool.run(
            tile_infos=all_tiles, batch_size=batch_size, result_dir=result_dir
        )

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs: list[tuple[int, "mp.Process"]] = []

    # Count active GPUs first to size the barrier correctly.
    active_gids = [gid for gid in gpu_ids if per_gpu.get(gid)]
    init_barrier = ctx.Barrier(len(active_gids)) if len(active_gids) > 1 else None

    for gid in active_gids:
        tiles = per_gpu[gid]
        p = ctx.Process(
            target=_gpu_worker_entry,
            args=(
                result_queue,
                gid,
                tiles,
                model_path,
                batch_size,
                result_dir,
                backend,
                io_workers,
                compress_npz,
                max_pixels,
                chunk_height,
                enforce_eager,
                init_barrier,
                adapter_path,
            ),
            daemon=False,
        )
        p.start()
        procs.append((gid, p))

    expected = len(procs)
    results: dict[int, dict] = {}
    while len(results) < expected:
        try:
            res = result_queue.get(timeout=5)
            results[res["gpu_id"]] = res
        except queue.Empty:
            if all(not p.is_alive() for _gid, p in procs):
                break

    for _gid, p in procs:
        p.join()

    errors: list[str] = []
    partial_paths: list[str] = []
    for gid, p in procs:
        res = results.get(gid)
        if res is None:
            msg = f"GPU {gid}: no result reported (exitcode={p.exitcode})"
            logger.error(msg)
            errors.append(msg)
            continue
        if res.get("error"):
            msg = f"GPU {gid} failed:\n{res['error']}"
            logger.error(msg)
            errors.append(msg)
            continue
        partial_path = res.get("partial_path", "")
        if partial_path:
            partial_paths.append(partial_path)
        if p.exitcode not in (0, None):
            errors.append(f"GPU {gid}: worker exited with code {p.exitcode}")

    if errors:
        logger.error("Multi-GPU worker failures:\n%s", "\n".join(errors))
        if not partial_paths:
            raise RuntimeError("All GPU workers failed:\n" + "\n".join(errors))
        logger.warning(
            "%d/%d GPU workers failed, continuing with %d partial results",
            len(errors),
            len(procs),
            len(partial_paths),
        )

    return partial_paths


# ---------------------------------------------------------------------------
# Multi-GPU orchestration
# ---------------------------------------------------------------------------


def embed_shard(
    shard_dir: str,
    output_dir: str,
    gpu_ids: list[int],
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 512,
    io_workers: int = 8,
    compress_npz: bool = False,
    reuse_workers: bool = False,
    skip_article_ids: set[int] | None = None,
    backend: str = "vllm",
    max_pixels: int | None = None,
    mode: str = "chunks",
    chunk_height: int | None = None,
    enforce_eager: bool = False,
    adapter_path: str | None = None,
) -> dict:
    """Scan a shard, embed across GPUs, merge results into a single .npz.

    Args:
        shard_dir: Path to shard directory.
        output_dir: Directory for output .npz file.
        gpu_ids: List of GPU IDs to use.
        model: Model name or path.
        batch_size: Tiles per embed call.
        skip_article_ids: Article IDs already embedded (for resume).
        backend: "vllm" or "sglang".

    Returns:
        Dict with keys: shard_dir, output_path, num_tiles, num_articles, elapsed_s.
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Extract shard_id from directory name
    shard_name = os.path.basename(shard_dir.rstrip("/"))  # e.g. "shard_042"
    try:
        shard_id = int(shard_name.split("_")[1])
    except (ValueError, IndexError):
        shard_id = 0

    # Scan images (chunks or tiles depending on mode)
    if mode == "chunks":
        tile_infos = scan_shard_chunks(shard_dir, skip_article_ids)
        unit = "chunks"
    else:
        tile_infos = scan_shard_tiles(shard_dir, skip_article_ids)
        unit = "tiles"
    if not tile_infos:
        logger.warning("No %s found in %s", unit, shard_dir)
        return {
            "shard_dir": shard_dir,
            "output_path": "",
            "num_tiles": 0,
            "num_articles": 0,
            "elapsed_s": time.time() - t0,
        }

    # Incremental: load existing .npz, detect new + updated chunks via mtime
    output_path = os.path.join(output_dir, f"shard_{shard_id:03d}.npz")
    existing_npz = None
    stale_keys: set[tuple[int, int, int]] = set()  # keys to replace in existing npz
    if os.path.exists(output_path):
        try:
            npz_mtime = os.path.getmtime(output_path)
            existing_npz = np.load(output_path)
            existing_keys = set(
                zip(
                    existing_npz["article_ids"].tolist(),
                    existing_npz["tile_indices"].tolist(),
                    existing_npz["chunk_indices"].tolist(),
                )
            )
            before = len(tile_infos)

            # Partition tile_infos into: new (not in npz) + updated (in npz but file is newer)
            new_infos = []
            for ti in tile_infos:
                key = (
                    ti.article_id,
                    ti.tile_index,
                    ti.chunk_index if hasattr(ti, "chunk_index") else 0,
                )
                if key not in existing_keys:
                    new_infos.append(ti)
                else:
                    # Check mtime — if chunk file is newer than npz, re-embed it
                    chunk_path = (
                        ti.chunk_path
                        if hasattr(ti, "chunk_path")
                        else getattr(ti, "tile_path", None)
                    )
                    if chunk_path and os.path.getmtime(chunk_path) > npz_mtime:
                        new_infos.append(ti)
                        stale_keys.add(key)

            tile_infos = new_infos
            logger.info(
                "Incremental: %d existing, %d new, %d updated (was %d total) in %s",
                len(existing_keys),
                len(tile_infos) - len(stale_keys),
                len(stale_keys),
                before,
                shard_name,
            )
            if not tile_infos:
                logger.info(
                    "Shard %d: all %d chunks up to date, skipping",
                    shard_id,
                    len(existing_keys),
                )
                return {
                    "shard_dir": shard_dir,
                    "output_path": output_path,
                    "num_tiles": len(existing_keys),
                    "num_articles": len(set(existing_npz["article_ids"].tolist())),
                    "elapsed_s": time.time() - t0,
                }
        except Exception as e:
            logger.warning(
                "Could not load existing %s for incremental: %s", output_path, e
            )
            existing_npz = None

    num_gpus = len(gpu_ids)
    if num_gpus == 1:
        # Single GPU — run in-process (no multiprocessing overhead)
        partial_path = gpu_worker(
            gpu_ids[0],
            tile_infos,
            model,
            batch_size,
            output_dir,
            backend,
            io_workers=io_workers,
            compress_npz=compress_npz,
            max_pixels=max_pixels,
            chunk_height=chunk_height,
            enforce_eager=enforce_eager,
            adapter_path=adapter_path,
        )
        partial_paths = [partial_path] if partial_path else []
    else:
        if reuse_workers:
            # Dynamic distribution — workers pull from shared queue
            logger.info(
                "Multi-GPU dynamic: %d tiles across %d GPUs", len(tile_infos), num_gpus
            )
            partial_paths = run_gpu_workers_parallel(
                per_gpu=None,
                gpu_ids=gpu_ids,
                model_path=model,
                batch_size=batch_size,
                result_dir=output_dir,
                backend=backend,
                io_workers=io_workers,
                compress_npz=compress_npz,
                reuse_workers=True,
                max_pixels=max_pixels,
                chunk_height=chunk_height,
                enforce_eager=enforce_eager,
                tile_infos=tile_infos,
                adapter_path=adapter_path,
            )
        else:
            # Static split — round-robin by article
            article_ids = sorted(set(t.article_id for t in tile_infos))
            gpu_assignment: dict[int, int] = {}
            for i, aid in enumerate(article_ids):
                gpu_assignment[aid] = gpu_ids[i % num_gpus]

            per_gpu: dict[int, list[TileInfo]] = {gid: [] for gid in gpu_ids}
            for ti in tile_infos:
                per_gpu[gpu_assignment[ti.article_id]].append(ti)

            for gid, tiles in per_gpu.items():
                logger.info(
                    "GPU %d: assigned %d tiles from %d articles",
                    gid,
                    len(tiles),
                    len({t.article_id for t in tiles}),
                )

            partial_paths = run_gpu_workers_parallel(
                per_gpu=per_gpu,
                gpu_ids=gpu_ids,
                model_path=model,
                batch_size=batch_size,
                result_dir=output_dir,
                backend=backend,
                io_workers=io_workers,
                compress_npz=compress_npz,
                reuse_workers=False,
                max_pixels=max_pixels,
                chunk_height=chunk_height,
                enforce_eager=enforce_eager,
                adapter_path=adapter_path,
            )

    # Merge partial results into final .npz
    if not partial_paths:
        logger.error("No partial results produced for %s", shard_dir)
        return {
            "shard_dir": shard_dir,
            "output_path": "",
            "num_tiles": 0,
            "num_articles": 0,
            "elapsed_s": time.time() - t0,
        }

    is_chunk_mode = mode == "chunks"
    all_emb, all_aids, all_tidx = [], [], []
    all_cidx, all_yo = [], []
    all_ph, all_vw, all_th = [], [], []
    all_ih, all_tp = [], []

    # Prepend existing data for incremental merge (excluding stale entries)
    if existing_npz is not None:
        if stale_keys:
            # Build mask: keep rows whose key is NOT in stale_keys
            ex_aids = existing_npz["article_ids"]
            ex_tidx = existing_npz["tile_indices"]
            ex_cidx = existing_npz["chunk_indices"]
            keep_mask = np.array(
                [
                    (int(ex_aids[j]), int(ex_tidx[j]), int(ex_cidx[j]))
                    not in stale_keys
                    for j in range(len(ex_aids))
                ],
                dtype=bool,
            )
            logger.info(
                "Incremental merge: keeping %d/%d existing rows (%d stale replaced)",
                keep_mask.sum(),
                len(keep_mask),
                len(stale_keys),
            )
            all_emb.append(existing_npz["embeddings"][keep_mask])
            all_aids.append(ex_aids[keep_mask])
            all_tidx.append(ex_tidx[keep_mask])
            all_cidx.append(ex_cidx[keep_mask])
            all_ph.append(existing_npz["page_heights"][keep_mask])
            all_vw.append(existing_npz["viewport_widths"][keep_mask])
            all_th.append(existing_npz["tile_heights"][keep_mask])
            all_ih.append(existing_npz["image_hashes"][keep_mask])
            all_tp.append(existing_npz["tile_paths"][keep_mask])
            if is_chunk_mode and "y_offsets" in existing_npz:
                all_yo.append(existing_npz["y_offsets"][keep_mask])
        else:
            all_emb.append(existing_npz["embeddings"])
            all_aids.append(existing_npz["article_ids"])
            all_tidx.append(existing_npz["tile_indices"])
            all_cidx.append(existing_npz["chunk_indices"])
            all_ph.append(existing_npz["page_heights"])
            all_vw.append(existing_npz["viewport_widths"])
            all_th.append(existing_npz["tile_heights"])
            all_ih.append(existing_npz["image_hashes"])
            all_tp.append(existing_npz["tile_paths"])
            if is_chunk_mode and "y_offsets" in existing_npz:
                all_yo.append(existing_npz["y_offsets"])

    # Deduplicate partial paths: persistent workers overwrite the same
    # partial_gpu{id}.npz for each work item, so the same file may appear
    # multiple times in partial_paths.  Reading it more than once duplicates
    # every embedding in the final .npz.
    seen_pp: set[str] = set()
    unique_partial_paths = []
    for pp in partial_paths:
        rp = os.path.realpath(pp)
        if rp not in seen_pp:
            seen_pp.add(rp)
            unique_partial_paths.append(pp)
    if len(unique_partial_paths) < len(partial_paths):
        logger.info(
            "Deduplicated partial paths: %d -> %d unique",
            len(partial_paths),
            len(unique_partial_paths),
        )

    for pp in unique_partial_paths:
        data = np.load(pp)
        all_emb.append(data["embeddings"])
        all_aids.append(data["article_ids"])
        all_tidx.append(data["tile_indices"])
        all_cidx.append(data["chunk_indices"])
        all_ph.append(data["page_heights"])
        all_vw.append(data["viewport_widths"])
        all_th.append(data["tile_heights"])
        all_ih.append(data["image_hashes"])
        all_tp.append(data["tile_paths"])
        if is_chunk_mode and "y_offsets" in data:
            all_yo.append(data["y_offsets"])

    embeddings = np.concatenate(all_emb, axis=0)
    article_ids = np.concatenate(all_aids)
    tile_indices = np.concatenate(all_tidx)
    chunk_indices = np.concatenate(all_cidx)
    page_heights = np.concatenate(all_ph)
    viewport_widths = np.concatenate(all_vw)
    tile_heights = np.concatenate(all_th)
    image_hashes = np.concatenate(all_ih)
    tile_paths = np.concatenate(all_tp)

    if is_chunk_mode:
        y_offsets = (
            np.concatenate(all_yo)
            if all_yo
            else np.zeros(len(embeddings), dtype=np.int32)
        )
        # Sort by (article_id, tile_index, chunk_index)
        sort_idx = np.lexsort((chunk_indices, tile_indices, article_ids))
    else:
        # Sort by (article_id, tile_index)
        sort_idx = np.lexsort((tile_indices, article_ids))

    embeddings = embeddings[sort_idx]
    article_ids = article_ids[sort_idx]
    tile_indices = tile_indices[sort_idx]
    chunk_indices = chunk_indices[sort_idx]
    page_heights = page_heights[sort_idx]
    viewport_widths = viewport_widths[sort_idx]
    tile_heights = tile_heights[sort_idx]
    image_hashes = image_hashes[sort_idx]
    tile_paths = tile_paths[sort_idx]

    extra_arrays = {}
    if is_chunk_mode:
        y_offsets = y_offsets[sort_idx]
        extra_arrays["y_offsets"] = y_offsets

    output_path = os.path.join(output_dir, f"shard_{shard_id:03d}.npz")
    save_npz(
        output_path,
        compressed=compress_npz,
        embeddings=embeddings,
        article_ids=article_ids,
        tile_indices=tile_indices,
        chunk_indices=chunk_indices,
        page_heights=page_heights,
        viewport_widths=viewport_widths,
        tile_heights=tile_heights,
        image_hashes=image_hashes,
        tile_paths=tile_paths,
        shard_id=np.int32(shard_id),
        **extra_arrays,
    )

    # Clean up partial files
    for pp in unique_partial_paths:
        try:
            os.remove(pp)
        except OSError:
            pass

    elapsed = time.time() - t0
    num_articles = len(set(article_ids.tolist()))
    logger.info(
        "Shard %d: %d embeddings (%d articles) in %.1fs -> %s",
        shard_id,
        len(embeddings),
        num_articles,
        elapsed,
        output_path,
    )

    return {
        "shard_dir": shard_dir,
        "output_path": output_path,
        "num_tiles": len(embeddings),
        "num_articles": num_articles,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Checkpoint for resume
# ---------------------------------------------------------------------------


def load_checkpoint(output_dir: str) -> set[int]:
    """Load set of already-embedded article IDs from checkpoint.json."""
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.exists(ckpt_path):
        return set()
    try:
        data = json.loads(Path(ckpt_path).read_text())
        return set(data.get("embedded_article_ids", []))
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", ckpt_path, e)
        return set()


def save_checkpoint(output_dir: str, article_ids: set[int]) -> None:
    """Save embedded article IDs to checkpoint.json."""
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    data = {"embedded_article_ids": sorted(article_ids)}
    Path(ckpt_path).write_text(json.dumps(data))
    logger.info("Checkpoint saved: %d article IDs -> %s", len(article_ids), ckpt_path)


# ---------------------------------------------------------------------------
# Patch mode: diff hashes, re-embed only changed tiles, update npz in-place
# ---------------------------------------------------------------------------


def _hash_file(path: str) -> str:
    """Compute MD5 hex digest of a file. Returns empty string on error."""
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def diff_shard_hashes(
    npz_path: str,
    shard_dir: str,
    hash_workers: int = 32,
) -> tuple[list[TileInfo], list[TileInfo], set[int]]:
    """Compare stored hashes in npz against current tile files on disk.

    Uses a thread pool to hash files in parallel (IO-bound on NVMe).

    Args:
        npz_path: Path to existing shard .npz file.
        shard_dir: Path to shard tile directory.
        hash_workers: Number of threads for parallel hashing.

    Returns:
        (stale_tiles, new_tiles, removed_rows):
            stale_tiles: TileInfos whose disk hash differs from stored hash.
            new_tiles: TileInfos on disk but not in npz at all.
            removed_rows: Row indices in npz whose tile_path no longer exists.
    """
    t0 = time.time()

    # Load stored index
    logger.info("Loading existing npz: %s", npz_path)
    data = np.load(npz_path)
    stored_paths = data["tile_paths"]  # S512
    stored_hashes = data["image_hashes"]  # S32
    logger.info(
        "Loaded %d stored embeddings in %.1fs", len(stored_paths), time.time() - t0
    )

    # Build index: tile_path -> (row_index, stored_hash)
    path_to_row: dict[str, tuple[int, str]] = {}
    for i in range(len(stored_paths)):
        p = (
            stored_paths[i].decode()
            if isinstance(stored_paths[i], bytes)
            else str(stored_paths[i])
        )
        h = (
            stored_hashes[i].decode()
            if isinstance(stored_hashes[i], bytes)
            else str(stored_hashes[i])
        )
        path_to_row[p] = (i, h)

    # Scan current tiles on disk
    disk_tiles = scan_shard_tiles(shard_dir)

    # Split into known (need hash check) vs new (not in npz)
    tiles_to_hash: list[TileInfo] = []
    new_tiles: list[TileInfo] = []
    seen_paths: set[str] = set()

    for ti in disk_tiles:
        seen_paths.add(ti.tile_path)
        if ti.tile_path in path_to_row:
            tiles_to_hash.append(ti)
        else:
            new_tiles.append(ti)

    # Parallel hash computation for known tiles
    stale_tiles: list[TileInfo] = []
    if tiles_to_hash:
        logger.info(
            "Hashing %d existing tiles (%d threads)...",
            len(tiles_to_hash),
            hash_workers,
        )
        t1 = time.time()
        matched = 0

        with ThreadPoolExecutor(max_workers=hash_workers) as pool:
            futures = {
                pool.submit(_hash_file, ti.tile_path): ti for ti in tiles_to_hash
            }
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Hashing tiles",
                unit="tile",
            )
            for fut in pbar:
                ti = futures[fut]
                current_hash = fut.result()
                _row_idx, old_hash = path_to_row[ti.tile_path]
                if current_hash and current_hash != old_hash:
                    stale_tiles.append(ti)
                else:
                    matched += 1
                pbar.set_postfix(stale=len(stale_tiles), matched=matched)
            pbar.close()

        hash_rate = len(tiles_to_hash) / max(time.time() - t1, 0.001)
        logger.info(
            "Hashed %d tiles in %.1fs (%.0f tiles/s): %d matched, %d stale",
            len(tiles_to_hash),
            time.time() - t1,
            hash_rate,
            matched,
            len(stale_tiles),
        )

    # Rows whose source file was deleted
    removed_rows: set[int] = set()
    for p, (row_idx, _h) in path_to_row.items():
        if p not in seen_paths:
            removed_rows.add(row_idx)

    elapsed = time.time() - t0
    logger.info(
        "Diff complete in %.1fs: %d stale, %d new, %d removed, %d unchanged "
        "(stored=%d, on_disk=%d)",
        elapsed,
        len(stale_tiles),
        len(new_tiles),
        len(removed_rows),
        len(tiles_to_hash) - len(stale_tiles),
        len(stored_paths),
        len(disk_tiles),
    )
    return stale_tiles, new_tiles, removed_rows


def patch_shard(
    npz_path: str,
    shard_dir: str,
    gpu_ids: list[int],
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 512,
    io_workers: int = 8,
    compress_npz: bool = False,
    yes: bool = False,
    dry_run: bool = False,
    backend: str = "vllm",
    max_pixels: int | None = None,
    adapter_path: str | None = None,
) -> dict:
    """Patch an existing shard npz by re-embedding only changed/new tiles.

    1. Diff stored hashes vs disk files
    2. Show summary and ask for confirmation
    3. Re-embed stale + new tiles
    4. Update npz: replace stale rows, append new rows, drop removed rows

    Args:
        npz_path: Path to existing shard .npz.
        shard_dir: Path to shard tile directory.
        gpu_ids: GPU IDs to use.
        model: Model name or path.
        batch_size: Tiles per embed() call.
        yes: Skip confirmation prompt.
        dry_run: Only diff, don't embed or write.

    Returns:
        Dict with keys: npz_path, patched, added, removed, unchanged.
    """
    stale_tiles, new_tiles, removed_rows = diff_shard_hashes(npz_path, shard_dir)

    tiles_to_embed = stale_tiles + new_tiles
    if not tiles_to_embed and not removed_rows:
        logger.info("Nothing to patch — all hashes match")
        return {
            "npz_path": npz_path,
            "patched": 0,
            "added": 0,
            "removed": 0,
            "unchanged": True,
        }

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Patch summary for {os.path.basename(npz_path)}")
    print(f"{'=' * 60}")
    print(f"  Stale (hash changed):  {len(stale_tiles):>8}  → re-embed")
    print(f"  New (not in npz):      {len(new_tiles):>8}  → embed")
    print(f"  Removed (file gone):   {len(removed_rows):>8}  → drop")
    print(f"  Total to embed:        {len(tiles_to_embed):>8}")
    print(f"{'=' * 60}\n")

    if dry_run:
        logger.info("Dry run — stopping before embedding")
        return {
            "npz_path": npz_path,
            "patched": len(stale_tiles),
            "added": len(new_tiles),
            "removed": len(removed_rows),
            "dry_run": True,
        }

    if not yes:
        answer = input("Proceed with patch? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            logger.info("Aborted by user")
            return {"npz_path": npz_path, "aborted": True}

    # Load existing data
    data = np.load(npz_path)
    old_embeddings = data["embeddings"]
    old_article_ids = data["article_ids"]
    old_tile_indices = data["tile_indices"]
    old_page_heights = data["page_heights"]
    old_viewport_widths = data["viewport_widths"]
    old_tile_heights = data["tile_heights"]
    old_image_hashes = data["image_hashes"]
    old_tile_paths = data["tile_paths"]
    shard_id = int(data["shard_id"])

    # Build stale path -> row index for replacement
    path_to_row: dict[str, int] = {}
    for i in range(len(old_tile_paths)):
        p = (
            old_tile_paths[i].decode()
            if isinstance(old_tile_paths[i], bytes)
            else str(old_tile_paths[i])
        )
        path_to_row[p] = i
    stale_row_indices = {path_to_row[ti.tile_path] for ti in stale_tiles}

    # Embed the changed + new tiles
    t_embed = time.time()
    if tiles_to_embed:
        logger.info(
            "Embedding %d tiles (%d stale + %d new) on GPUs %s",
            len(tiles_to_embed),
            len(stale_tiles),
            len(new_tiles),
            gpu_ids,
        )
        output_dir = os.path.dirname(npz_path)

        num_gpus = len(gpu_ids)
        if num_gpus == 1:
            partial_path = gpu_worker(
                gpu_ids[0],
                tiles_to_embed,
                model,
                batch_size,
                output_dir,
                backend,
                io_workers=io_workers,
                compress_npz=compress_npz,
                max_pixels=max_pixels,
                adapter_path=adapter_path,
            )
            partial_paths = [partial_path] if partial_path else []
        else:
            article_ids_list = sorted(set(t.article_id for t in tiles_to_embed))
            gpu_assignment = {
                aid: gpu_ids[i % num_gpus] for i, aid in enumerate(article_ids_list)
            }
            per_gpu: dict[int, list[TileInfo]] = {gid: [] for gid in gpu_ids}
            for ti in tiles_to_embed:
                per_gpu[gpu_assignment[ti.article_id]].append(ti)

            partial_paths = run_gpu_workers_parallel(
                per_gpu=per_gpu,
                gpu_ids=gpu_ids,
                model_path=model,
                batch_size=batch_size,
                result_dir=output_dir,
                backend=backend,
                io_workers=io_workers,
                compress_npz=compress_npz,
                max_pixels=max_pixels,
                adapter_path=adapter_path,
            )

        # Collect new embeddings
        new_emb, new_aids, new_tidx = [], [], []
        new_ph, new_vw, new_th = [], [], []
        new_ih, new_tp = [], []
        for pp in partial_paths:
            d = np.load(pp)
            new_emb.append(d["embeddings"])
            new_aids.append(d["article_ids"])
            new_tidx.append(d["tile_indices"])
            new_ph.append(d["page_heights"])
            new_vw.append(d["viewport_widths"])
            new_th.append(d["tile_heights"])
            new_ih.append(d["image_hashes"])
            new_tp.append(d["tile_paths"])
            try:
                os.remove(pp)
            except OSError:
                pass

        if new_emb:
            fresh_embeddings = np.concatenate(new_emb, axis=0)
            fresh_article_ids = np.concatenate(new_aids)
            fresh_tile_indices = np.concatenate(new_tidx)
            fresh_page_heights = np.concatenate(new_ph)
            fresh_viewport_widths = np.concatenate(new_vw)
            fresh_tile_heights = np.concatenate(new_th)
            fresh_image_hashes = np.concatenate(new_ih)
            fresh_tile_paths = np.concatenate(new_tp)
        else:
            tiles_to_embed = []  # embedding failed

    # Build path index for fresh results
    fresh_by_path: dict[str, int] = {}
    if tiles_to_embed:
        for i in range(len(fresh_tile_paths)):
            p = (
                fresh_tile_paths[i].decode()
                if isinstance(fresh_tile_paths[i], bytes)
                else str(fresh_tile_paths[i])
            )
            fresh_by_path[p] = i

    # Assemble final arrays: keep unchanged rows, replace stale, drop removed
    rows_to_drop = stale_row_indices | removed_rows
    keep_mask = np.ones(len(old_embeddings), dtype=bool)
    for r in rows_to_drop:
        keep_mask[r] = False

    parts_emb = [old_embeddings[keep_mask]]
    parts_aids = [old_article_ids[keep_mask]]
    parts_tidx = [old_tile_indices[keep_mask]]
    parts_ph = [old_page_heights[keep_mask]]
    parts_vw = [old_viewport_widths[keep_mask]]
    parts_th = [old_tile_heights[keep_mask]]
    parts_ih = [old_image_hashes[keep_mask]]
    parts_tp = [old_tile_paths[keep_mask]]

    if tiles_to_embed:
        parts_emb.append(fresh_embeddings)
        parts_aids.append(fresh_article_ids)
        parts_tidx.append(fresh_tile_indices)
        parts_ph.append(fresh_page_heights)
        parts_vw.append(fresh_viewport_widths)
        parts_th.append(fresh_tile_heights)
        parts_ih.append(fresh_image_hashes)
        parts_tp.append(fresh_tile_paths)

    final_emb = np.concatenate(parts_emb, axis=0)
    final_aids = np.concatenate(parts_aids)
    final_tidx = np.concatenate(parts_tidx)
    final_ph = np.concatenate(parts_ph)
    final_vw = np.concatenate(parts_vw)
    final_th = np.concatenate(parts_th)
    final_ih = np.concatenate(parts_ih)
    final_tp = np.concatenate(parts_tp)

    # Re-sort
    sort_idx = np.lexsort((final_tidx, final_aids))
    final_emb = final_emb[sort_idx]
    final_aids = final_aids[sort_idx]
    final_tidx = final_tidx[sort_idx]
    final_ph = final_ph[sort_idx]
    final_vw = final_vw[sort_idx]
    final_th = final_th[sort_idx]
    final_ih = final_ih[sort_idx]
    final_tp = final_tp[sort_idx]

    save_npz(
        npz_path,
        compressed=compress_npz,
        embeddings=final_emb,
        article_ids=final_aids,
        tile_indices=final_tidx,
        page_heights=final_ph,
        viewport_widths=final_vw,
        tile_heights=final_th,
        image_hashes=final_ih,
        tile_paths=final_tp,
        shard_id=np.int32(shard_id),
    )

    elapsed_total = time.time() - t_embed
    logger.info(
        "Patched %s in %.1fs: %d replaced, %d added, %d removed, %d total",
        npz_path,
        elapsed_total,
        len(stale_tiles),
        len(new_tiles),
        len(removed_rows),
        len(final_emb),
    )
    return {
        "npz_path": npz_path,
        "patched": len(stale_tiles),
        "added": len(new_tiles),
        "removed": len(removed_rows),
        "total": len(final_emb),
        "unchanged": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed tiles from a single shard using Qwen3-VL-Embedding-2B.",
    )
    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Path to shard directory (e.g. output_coordinated/shard_042)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for .npz file"
    )
    parser.add_argument(
        "--gpu-ids",
        default="all",
        help="Comma-separated GPU IDs, or 'all' (default: all)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Model name or path (default: Qwen/Qwen3-VL-Embedding-2B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tiles per embed() call (default: 128)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=8,
        help="Threads per GPU process for tile read/hash/decode (default: 8)",
    )
    parser.add_argument(
        "--compress-npz",
        action="store_true",
        help="Compress output npz files (smaller but slower)",
    )
    parser.add_argument(
        "--reuse-workers",
        action="store_true",
        help="Reuse persistent GPU workers across multiple embed_shard calls in-process",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "sglang", "direct_gpu"],
        default="sglang",
        help="Embedding backend (default: sglang)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already embedded articles)",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Patch mode: diff hashes, re-embed only changed tiles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --patch: only show diff, don't embed or write",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="With --patch: skip confirmation prompt",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=32,
        help="Threads for parallel hashing in patch mode (default: 32)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Max pixels per tile image before resize (e.g. 200000). "
        "Reduces visual tokens for faster inference. None = no resize.",
    )
    parser.add_argument(
        "--chunk-height",
        type=int,
        default=None,
        help="Split tall tiles into chunks of this height (e.g. 1024). "
        "Each chunk gets a separate embedding. None = no chunking.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture in vLLM (fixes hangs on some GPUs).",
    )
    parser.add_argument(
        "--redirects-json",
        default=None,
        help="Path to .redirects.json to skip redirect articles",
    )
    parser.add_argument(
        "--mode",
        choices=["chunks", "tiles"],
        default="chunks",
        help="Embedding unit: 'chunks' (1024px strips, default) or 'tiles' (full 8192px tiles)",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="System prompt instruction for embedding (default: %(default)r)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to PEFT LoRA adapter checkpoint directory. "
        "Loaded and merged into base model weights before embedding. "
        "Only supported with direct_gpu backend.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = resolve_gpu_ids(args.gpu_ids)

    if args.adapter:
        if args.backend != "direct_gpu":
            logger.info(
                "--adapter requires direct_gpu backend, overriding --backend=%s",
                args.backend,
            )
            args.backend = "direct_gpu"

    if args.instruction != DEFAULT_INSTRUCTION:
        set_instruction(args.instruction)

    if args.patch:
        # Patch mode: find existing npz, diff, re-embed changed tiles
        shard_name = os.path.basename(args.shard_dir.rstrip("/"))
        try:
            shard_id = int(shard_name.split("_")[1])
        except (ValueError, IndexError):
            shard_id = 0
        npz_path = os.path.join(args.output_dir, f"shard_{shard_id:03d}.npz")
        if not os.path.exists(npz_path):
            logger.error("Patch mode requires existing npz at %s", npz_path)
            return

        logger.info("Patch mode: %s vs %s", npz_path, args.shard_dir)
        result = patch_shard(
            npz_path=npz_path,
            shard_dir=args.shard_dir,
            gpu_ids=gpu_ids,
            model=args.model,
            batch_size=args.batch_size,
            io_workers=args.io_workers,
            compress_npz=args.compress_npz,
            yes=args.yes,
            dry_run=args.dry_run,
            backend=args.backend,
            max_pixels=args.max_pixels,
            adapter_path=args.adapter,
        )
        logger.info("Patch done: %s", result)
        return

    # Normal mode
    logger.info("Embedding shard %s on GPUs %s", args.shard_dir, gpu_ids)

    skip_ids = load_checkpoint(args.output_dir) if args.resume else set()
    if skip_ids:
        logger.info("Resuming: skipping %d already-embedded articles", len(skip_ids))

    if args.redirects_json:
        redirect_ids = load_redirect_ids(args.redirects_json)
        skip_ids |= redirect_ids
        logger.info("After adding redirects: skipping %d total articles", len(skip_ids))

    result = embed_shard(
        shard_dir=args.shard_dir,
        output_dir=args.output_dir,
        gpu_ids=gpu_ids,
        model=args.model,
        batch_size=args.batch_size,
        io_workers=args.io_workers,
        compress_npz=args.compress_npz,
        reuse_workers=args.reuse_workers,
        skip_article_ids=skip_ids,
        backend=args.backend,
        max_pixels=args.max_pixels,
        mode=args.mode,
        chunk_height=args.chunk_height,
        enforce_eager=args.enforce_eager,
        adapter_path=args.adapter,
    )

    if result["num_tiles"] > 0:
        # Update checkpoint with newly embedded articles
        npz = np.load(result["output_path"])
        new_ids = set(npz["article_ids"].tolist())
        all_ids = skip_ids | new_ids
        save_checkpoint(args.output_dir, all_ids)

    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
