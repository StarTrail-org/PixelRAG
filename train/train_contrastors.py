#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-Embedding with contrastors-style GradCache training.

Uses contrastors' core ideas (GradCache, distributed gather_with_grad) with
colpali-engine's BiQwen3 model + processor for Qwen3-VL vision-language support.

Single GPU:
    CUDA_VISIBLE_DEVICES=3 python training/train_contrastors.py --max-steps 100

Multi-GPU (5 GPUs, cross-GPU negatives + GradCache):
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nproc_per_node=5 \
        training/train_contrastors.py --max-steps 500

Resume:
    python training/train_contrastors.py --resume training/output_contrastors/checkpoint-200
"""

import argparse
import datetime
import functools
import json
import logging
import os
import signal
import io
import sys
import time
import urllib.parse
from urllib.parse import unquote
from pathlib import Path
from contextlib import nullcontext
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated images in external datasets
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrastors core: GradCache + distributed gather (adapted from nomic-ai/contrastors)
# ---------------------------------------------------------------------------


class RandContext:
    """Save/restore RNG state for GradCache replay (from contrastors/rand_state.py)."""

    def __init__(self, tensors):
        if isinstance(tensors, dict):
            tensors = list(tensors.values())
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def gather_with_grad(t):
    """All-gather tensors across GPUs while preserving gradients."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    if t.ndim == 0:
        t = t.unsqueeze(0)
    return torch.cat(torch.distributed.nn.all_gather(t), dim=0)


class LogitScale(torch.nn.Module):
    """Learnable logit scale (from contrastors/OpenCLIP).

    Contrastors uses a learnable log-scale initialized to ln(1/0.07) ≈ 2.66.
    The parameter is clamped in-place AFTER optimizer.step() (not in forward),
    matching contrastors' approach. Clamping in forward would create a dead zone
    where gradients are zero when log_scale exceeds the limit.
    """

    def __init__(self, init_value=1 / 0.07, max_value=100.0):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.log(torch.tensor(init_value)))
        self.max_log = float(torch.log(torch.tensor(max_value)))

    def forward(self, x):
        return x * self.log_scale.exp()

    @torch.no_grad()
    def clamp_(self):
        """Clamp log_scale in-place. Call after optimizer.step()."""
        self.log_scale.clamp_(0, self.max_log)


class EMAModel:
    """Exponential Moving Average of model parameters for better generalization."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA params. Call before eval."""
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        """Restore original params after eval."""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def siglip_loss(query, document, logit_scale, gather_enabled=False, hardness_alpha=0.0):
    """SigLIP pairwise sigmoid loss (Zhai et al. 2023).

    Instead of softmax cross-entropy over the row, applies binary sigmoid loss
    to each (query, document) pair independently. Positive pairs should have high
    similarity, all other pairs should have low similarity.

    This avoids the softmax bottleneck where increasing one logit requires
    decreasing all others — each pair is classified independently.
    """
    if gather_enabled:
        document = gather_with_grad(document)

    device = query.device
    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    num_queries = query.shape[0]
    num_docs = document.shape[0]

    # Compute scaled similarity matrix
    similarity = logit_scale(torch.matmul(query, document.T))

    # Build target matrix: +1 for positive pairs, -1 for negatives
    if gather_enabled and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Compute positive indices (same logic as clip_loss for hard negatives)
    labels = torch.arange(num_queries, device=device) + rank * num_queries
    assert num_docs % (num_queries * world_size) == 0
    stride = num_docs // (num_queries * world_size)
    pos_indices = labels * stride  # shape: [num_queries]

    # Target: -1 everywhere, +1 at positive positions
    target = -torch.ones(num_queries, num_docs, device=device, dtype=query.dtype)
    target.scatter_(1, pos_indices.unsqueeze(1).long(), 1.0)

    # SigLIP loss: -log_sigmoid(target * similarity) averaged over all pairs
    loss = -F.logsigmoid(target * similarity).sum() / num_queries
    if gather_enabled and dist.is_initialized():
        loss = loss * dist.get_world_size()

    # Accuracy: check if positive has highest similarity
    accuracy = (similarity.argmax(dim=1) == pos_indices).float().mean()
    return loss, accuracy


def clip_loss(
    query,
    document,
    logit_scale,
    gather_enabled=False,
    hardness_alpha=0.0,
    label_smoothing=0.0,
):
    """InfoNCE contrastive loss with hard negative support (from contrastors/loss.py).

    Inspired by: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L66

    When hard negatives are used, document contains interleaved positives and negatives:
        [pos1, neg1a, neg1b, pos2, neg2a, neg2b, ...]
    So document.size(0) = query.size(0) * (1 + num_hard_negatives).
    Labels point to the positive positions: [0, 3, 6, 9, ...] for 2 hard negs.

    hardness_alpha: LLaVE-style hardness weighting (Lan et al. 2025). When > 0,
    adds alpha * cos_sim(q, d) to negative logits before softmax, so harder
    negatives get upweighted. 0 = off (standard InfoNCE).
    """
    if gather_enabled:
        document = gather_with_grad(document)

    device = query.device

    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    num_queries = query.shape[0]
    labels = torch.arange(num_queries).to(device)

    similarity = logit_scale(torch.matmul(query, document.T))

    # Rank offset and world_size scaling only apply when documents are gathered
    # across ranks. Without gather, each rank evaluates independently (local labels).
    if gather_enabled and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    labels = labels + rank * num_queries
    # Scale for hard negatives: docs_per_query_per_rank = doc.size(0) / (N * W)
    assert document.size(0) % (num_queries * world_size) == 0, (
        f"document.size(0)={document.size(0)} not divisible by "
        f"num_queries*world_size={num_queries}*{world_size}"
    )
    labels = labels * (document.size(0) // (num_queries * world_size))

    # LLaVE hardness weighting: upweight harder negatives in softmax
    if hardness_alpha > 0:
        with torch.no_grad():
            raw_sim = torch.matmul(query, document.T)
        neg_mask = torch.ones_like(similarity, dtype=torch.bool)
        neg_mask.scatter_(1, labels.unsqueeze(1), False)
        similarity = similarity + hardness_alpha * raw_sim * neg_mask

    loss = F.cross_entropy(similarity, labels, label_smoothing=label_smoothing)
    if gather_enabled and dist.is_initialized():
        loss = loss * dist.get_world_size()

    # accuracy for logging
    accuracy = (similarity.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


def get_chunked_embeddings(model, chunks, process_fn):
    """Forward pass on chunks without grad, caching embeddings + RNG states."""
    embeddings = []
    rand_states = []
    # Use the unwrapped model for no-grad forward (avoids DDP overhead)
    raw_model = model.module if hasattr(model, "module") else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for chunk in chunks:
                rand_states.append(RandContext(chunk))
                emb = process_fn(raw_model, chunk)
                embeddings.append(emb)
    return torch.cat(embeddings, dim=0), rand_states


def grad_cache_loss(
    model,
    query_chunks,
    doc_chunks,
    logit_scale,
    query_process_fn,
    doc_process_fn,
    hardness_alpha=0.0,
    loss_fn=None,
):
    """GradCache: large effective batch with constant memory (from contrastors/loss.py).

    1. Forward all chunks WITHOUT grad → cache embeddings
    2. Compute loss on full gathered embeddings → cache gradients
    3. Replay forward WITH grad, backprop through surrogate loss

    All surrogate backward passes run under no_sync(), then we manually
    all_reduce gradients for both the model and logit_scale.
    """
    if loss_fn is None:
        loss_fn = clip_loss
    raw_model = model.module if hasattr(model, "module") else model

    # Step 1: get all embeddings without grad (no DDP involvement)
    query_embs, query_states = get_chunked_embeddings(
        model, query_chunks, query_process_fn
    )
    doc_embs, doc_states = get_chunked_embeddings(model, doc_chunks, doc_process_fn)

    # Step 2: compute loss, get gradient w.r.t. embeddings
    # This uses all_gather (NCCL collective #1) and its backward does
    # reduce_scatter (NCCL collective #2). Both happen on detached tensors,
    # NOT through DDP, so DDP doesn't interfere.
    query_embs_d = query_embs.detach().requires_grad_()
    doc_embs_d = doc_embs.detach().requires_grad_()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, accuracy = loss_fn(
            query_embs_d,
            doc_embs_d,
            logit_scale,
            gather_enabled=True,
            hardness_alpha=hardness_alpha,
        )
        loss.backward()

    query_cache = query_embs_d.grad
    doc_cache = doc_embs_d.grad
    loss_val = loss.detach()

    # Step 3: accumulate real gradients via surrogate loss
    # Combine all chunks into a single list so DDP sync happens exactly ONCE
    # (on the very last backward call). This matches contrastors' pattern.
    # Split grad cache to match the actual chunk sizes (doc may differ from query
    # when hard negatives are used — doc batch = query_batch * (1+num_hard_neg))
    q_chunk_sizes = [c["input_ids"].shape[0] for c in query_chunks]
    d_chunk_sizes = [c["input_ids"].shape[0] for c in doc_chunks]
    query_grad_chunks = query_cache.split(q_chunk_sizes)
    doc_grad_chunks = doc_cache.split(d_chunk_sizes)

    all_chunks = list(query_chunks) + list(doc_chunks)
    all_grads = list(query_grad_chunks) + list(doc_grad_chunks)
    all_states = list(query_states) + list(doc_states)
    all_fns = [query_process_fn] * len(query_chunks) + [doc_process_fn] * len(
        doc_chunks
    )

    # All backward calls use no_sync to avoid DDP reducer confusion (query chunks
    # skip visual encoder while doc chunks use it → different "used" parameter sets
    # across chunks causes intermittent NCCL deadlocks with find_unused_parameters).
    # We manually all_reduce gradients after all backward calls.
    has_ddp = hasattr(model, "no_sync")
    for chunk, grad, state, fn in zip(all_chunks, all_grads, all_states, all_fns):
        sync_ctx = model.no_sync() if has_ddp else nullcontext()
        with sync_ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with state:
                    emb = fn(raw_model, chunk)
                surrogate = torch.dot(emb.flatten(), grad.flatten())
            surrogate.backward()

    # Manual gradient all_reduce (replaces DDP's built-in sync).
    # Always call all_reduce on every requires_grad param (fill zero grad if None)
    # to ensure all ranks issue the same number of NCCL collectives — prevents
    # deadlocks if different chunks produce gradients on different param subsets.
    if has_ddp and dist.is_initialized():
        manual_all_reduce_grads(model, logit_scale)

    return loss_val, accuracy.detach()


def manual_all_reduce_grads(*modules):
    """Average gradients across ranks for modules outside DDP's reducer path."""
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def debug_trace(enabled, message):
    """Emit a per-rank debug line when tracing distributed hangs."""
    if not enabled:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(f"[debug rank={rank}] {message}")


def build_openai_client(api_base_url="", timeout=60):
    """Create an OpenAI-compatible client for local or hosted endpoints."""
    from openai import OpenAI  # pyright: ignore[reportMissingImports]

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_base_url:
        # Local vLLM endpoints use their own auth; don't leak OPENAI_API_KEY
        vllm_key = os.environ.get("VLLM_API_KEY", "dummy")
        return OpenAI(base_url=api_base_url, api_key=vllm_key, timeout=timeout)
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide an OpenAI key or pass --vllm-url "
            "for a local OpenAI-compatible endpoint."
        )
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base, timeout=timeout)


def preflight_simpleqa_client(vllm_url="", model_name=""):
    """Check SimpleQA answer/judge client before expensive model loading."""
    try:
        client = build_openai_client(vllm_url, timeout=30)
        client.models.list()
        target = vllm_url or "openai"
        logger.info(f"SimpleQA API preflight OK: target={target} model={model_name}")
        return True
    except Exception as e:
        target = vllm_url or "openai"
        logger.warning(f"SimpleQA API preflight failed for {target}: {e}")
        return False


def grad_cache_loss_query_side(
    query_model,
    frozen_doc_model,
    query_chunks,
    doc_chunks,
    logit_scale,
    query_process_fn,
    doc_process_fn,
    debug_enabled=False,
    hardness_alpha=0.0,
):
    """GradCache variant for query-only tuning with a frozen document tower."""
    raw_query_model = (
        query_model.module if hasattr(query_model, "module") else query_model
    )

    # Step 1: query embeddings come from the trainable tower; document embeddings
    # come from the frozen base tower so the datastore stays valid.
    debug_trace(debug_enabled, "gradcache_query_side: start query no-grad pass")
    query_embs, query_states = get_chunked_embeddings(
        query_model, query_chunks, query_process_fn
    )

    debug_trace(debug_enabled, "gradcache_query_side: start doc frozen pass")
    doc_embs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for chunk in doc_chunks:
                doc_embs.append(doc_process_fn(frozen_doc_model, chunk))
    doc_embs = torch.cat(doc_embs, dim=0)

    # Step 2: compute loss and cache only the query-side gradients.
    debug_trace(
        debug_enabled, "gradcache_query_side: start loss backward on detached embs"
    )
    query_embs_d = query_embs.detach().requires_grad_()
    doc_embs_d = doc_embs.detach().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, accuracy = clip_loss(
            query_embs_d,
            doc_embs_d,
            logit_scale,
            gather_enabled=True,
            hardness_alpha=hardness_alpha,
        )
        loss.backward()

    query_cache = query_embs_d.grad
    loss_val = loss.detach()

    # Step 3: replay only query chunks; doc tower is frozen and never backprops.
    q_chunk_sizes = [c["input_ids"].shape[0] for c in query_chunks]
    query_grad_chunks = query_cache.split(q_chunk_sizes)

    has_ddp = hasattr(query_model, "no_sync")
    debug_trace(debug_enabled, "gradcache_query_side: replay query chunks")
    for chunk, grad, state in zip(query_chunks, query_grad_chunks, query_states):
        sync_ctx = query_model.no_sync() if has_ddp else nullcontext()
        with sync_ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with state:
                    emb = query_process_fn(raw_query_model, chunk)
                surrogate = torch.dot(emb.flatten(), grad.flatten())
            surrogate.backward()

    if has_ddp and dist.is_initialized():
        debug_trace(debug_enabled, "gradcache_query_side: manual all_reduce grads")
        manual_all_reduce_grads(query_model, logit_scale)
    debug_trace(debug_enabled, "gradcache_query_side: done")

    return loss_val, accuracy.detach()


def direct_loss_query_side(
    query_model,
    frozen_doc_model,
    q_inputs,
    d_inputs,
    logit_scale,
    gather_enabled=True,
    debug_enabled=False,
    hardness_alpha=0.0,
):
    """Simpler query-side training path without GradCache."""
    raw_query_model = (
        query_model.module if hasattr(query_model, "module") else query_model
    )
    debug_trace(debug_enabled, "direct_query_side: query forward")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _clear_rope_deltas(raw_query_model)
        q_emb = raw_query_model(**q_inputs)
    debug_trace(debug_enabled, "direct_query_side: doc frozen forward")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            _clear_rope_deltas(frozen_doc_model)
            d_emb = frozen_doc_model(**d_inputs)
    debug_trace(debug_enabled, "direct_query_side: loss backward")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, accuracy = clip_loss(
            q_emb,
            d_emb,
            logit_scale,
            gather_enabled=gather_enabled,
            hardness_alpha=hardness_alpha,
        )
        loss.backward()
    if dist.is_initialized():
        debug_trace(debug_enabled, "direct_query_side: manual all_reduce grads")
        manual_all_reduce_grads(query_model, logit_scale)
    debug_trace(debug_enabled, "direct_query_side: done")
    return loss.detach(), accuracy.detach()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class QueryImageDataset(Dataset):
    """Dataset of (query, positive_path, [negative_paths...]) from JSONL.

    Supports two formats:
    - Simple: {"query": "...", "chunk_path": "..."}
    - Hard negatives: {"query": "...", "chunk_path": "...", "neg_chunk_paths": ["...", ...]}

    Pre-validates all images at init time so every __getitem__ is guaranteed
    to succeed — critical for DDP where all ranks must process the same
    number of batches (a None/skipped batch on one rank deadlocks NCCL).
    """

    def __init__(
        self,
        jsonl_path,
        max_pairs=None,
        num_hard_negatives=0,
        skip_image_verify=False,
        reverse=False,
    ):
        self.pairs = []  # (query, pos_path, [neg_path1, neg_path2, ...])
        self.num_hard_negatives = num_hard_negatives
        jsonl_dir = Path(jsonl_path).resolve().parent
        skipped = 0

        def _resolve_path(path_str):
            path = Path(path_str)
            if path.is_absolute():
                return str(path)
            return str((jsonl_dir / path).resolve())

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                pos_path = _resolve_path(item["chunk_path"])
                if not os.path.exists(pos_path):
                    skipped += 1
                    continue
                if not skip_image_verify:
                    try:
                        with Image.open(pos_path) as im:
                            im.convert("RGB").verify()
                    except Exception as e:
                        logger.warning(f"Bad image {pos_path}: {e}")
                        skipped += 1
                        continue

                # Collect hard negatives if present (path-check only, no image verify)
                neg_paths = []
                if num_hard_negatives > 0 and "neg_chunk_paths" in item:
                    for np_ in item["neg_chunk_paths"][:num_hard_negatives]:
                        np_ = _resolve_path(np_)
                        if os.path.exists(np_):
                            neg_paths.append(np_)
                    # Pad with None if not enough negatives (will be skipped in collate)
                    while len(neg_paths) < num_hard_negatives:
                        neg_paths.append(None)

                self.pairs.append((item["query"], pos_path, neg_paths))
        if reverse:
            self.pairs.reverse()
            logger.info("Reversed data order (high-quality-first curriculum)")
        if max_pairs:
            self.pairs = self.pairs[:max_pairs]
        if skipped:
            logger.warning(f"Skipped {skipped} missing/bad images at init")
        n_with_negs = sum(
            1 for _, _, negs in self.pairs if any(n is not None for n in negs)
        )
        logger.info(
            f"Loaded {len(self.pairs)} valid pairs from {jsonl_path}"
            + (
                f" ({n_with_negs} with hard negatives)"
                if num_hard_negatives > 0
                else ""
            )
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class MixedBatchSampler(Sampler):
    """Yields batches with exact per-source counts from a ConcatDataset.

    Each source occupies a contiguous index range [start, start+length).
    Every batch draws exactly `count` samples from each source.
    Sources cycle (reshuffle) when exhausted.

    Args:
        source_ranges: list of (start_idx, dataset_length, count_per_batch)
        shuffle: shuffle indices within each source per epoch
        seed: base random seed
        rank/world_size: for distributed training (each rank gets disjoint batches)
    """

    def __init__(self, source_ranges, shuffle=True, seed=42, rank=0, world_size=1):
        self.source_ranges = source_ranges
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._num_batches = min(
            length // count for _, length, count in source_ranges if count > 0
        )
        # In distributed mode, each rank gets a subset of batches
        self._num_batches = self._num_batches // world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        # Build shuffled index pools per source
        pools = []
        for start, length, count in self.source_ranges:
            if self.shuffle:
                perm = torch.randperm(length, generator=rng).tolist()
            else:
                perm = list(range(length))
            needed = (self._num_batches * self.world_size + 1) * count
            full = (perm * ((needed // length) + 1))[:needed]
            pools.append((start, full, count))
        # Yield batches for this rank
        total_batches = self._num_batches * self.world_size
        for b in range(total_batches):
            if b % self.world_size != self.rank:
                continue
            batch = []
            for start, pool, count in pools:
                offset = b * count
                batch.extend(start + pool[offset + i] for i in range(count))
            yield batch

    def __len__(self):
        return self._num_batches


class TextQueryDataset(Dataset):
    """Dataset of (query, positive_text, [negative_texts...]) from text-qa-pair JSONL.

    Each row has: query, passage (positive), neg_passages (list of hard negative texts).
    """

    def __init__(self, jsonl_paths, max_pairs=None, num_hard_negatives=0):
        self.pairs = []
        self.num_hard_negatives = num_hard_negatives
        skipped = 0

        for jsonl_path in (
            jsonl_paths if isinstance(jsonl_paths, list) else [jsonl_paths]
        ):
            with open(jsonl_path) as f:
                for line in f:
                    item = json.loads(line)
                    if "passage" not in item or not item["passage"]:
                        skipped += 1
                        continue

                    neg_texts = []
                    if num_hard_negatives > 0 and "neg_passages" in item:
                        for nt in item["neg_passages"][:num_hard_negatives]:
                            if nt:
                                neg_texts.append(nt)
                        while len(neg_texts) < num_hard_negatives:
                            neg_texts.append(item["passage"])  # pad with positive

                    self.pairs.append((item["query"], item["passage"], neg_texts))
                    if max_pairs and len(self.pairs) >= max_pairs:
                        break
            if max_pairs and len(self.pairs) >= max_pairs:
                break

        logger.info(
            f"Loaded {len(self.pairs)} text pairs from {jsonl_paths} "
            f"(skipped {skipped})"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# Task-specific instructions (Qwen3-VL-Embedding style)
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."

# Pre-computed chat template strings (filled by init_chat_templates).
_QUERY_PREFIX = None  # e.g. "<|im_start|>system\n...query instruction...<|im_end|>\n<|im_start|>user\n"
_QUERY_SUFFIX = None  # e.g. "<|im_end|>\n<|im_start|>assistant\n"
_DOC_IMAGE_TMPL = None  # static template for image docs
_DOC_TEXT_PREFIX = None  # prefix for text doc template
_DOC_TEXT_SUFFIX = None  # suffix for text doc template


def init_chat_templates(processor):
    """Pre-compute chat template strings once to avoid per-batch overhead."""
    global \
        _QUERY_PREFIX, \
        _QUERY_SUFFIX, \
        _DOC_IMAGE_TMPL, \
        _DOC_TEXT_PREFIX, \
        _DOC_TEXT_SUFFIX
    # Query template with placeholder
    q_msgs = [
        {"role": "system", "content": [{"type": "text", "text": QUERY_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": "PLACEHOLDER"}]},
    ]
    q_text = processor.apply_chat_template(
        q_msgs, tokenize=False, add_generation_prompt=True
    )
    idx = q_text.index("PLACEHOLDER")
    _QUERY_PREFIX = q_text[:idx]
    _QUERY_SUFFIX = q_text[idx + len("PLACEHOLDER") :]
    # Image doc template is fully static (no per-sample text)
    d_msgs = [
        {"role": "system", "content": [{"type": "text", "text": DOC_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    _DOC_IMAGE_TMPL = processor.apply_chat_template(
        d_msgs, tokenize=False, add_generation_prompt=True
    )
    # Text doc template with placeholder (for text-only training)
    dt_msgs = [
        {"role": "system", "content": [{"type": "text", "text": DOC_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": "PLACEHOLDER"}]},
    ]
    dt_text = processor.apply_chat_template(
        dt_msgs, tokenize=False, add_generation_prompt=True
    )
    dt_idx = dt_text.index("PLACEHOLDER")
    _DOC_TEXT_PREFIX = dt_text[:dt_idx]
    _DOC_TEXT_SUFFIX = dt_text[dt_idx + len("PLACEHOLDER") :]
    logger.info(f"Query prefix: {repr(_QUERY_PREFIX)}")
    logger.info(f"Doc image template: {repr(_DOC_IMAGE_TMPL[:80])}...")
    logger.info(f"Doc text prefix: {repr(_DOC_TEXT_PREFIX)}")


def process_queries(processor, queries):
    """Wrap queries in chat template with retrieval instruction."""
    texts = [_QUERY_PREFIX + q + _QUERY_SUFFIX for q in queries]
    return processor(text=texts, return_tensors="pt", padding="longest")


def process_doc_texts(processor, texts):
    """Wrap document texts in chat template with representation instruction."""
    wrapped = [_DOC_TEXT_PREFIX + t + _DOC_TEXT_SUFFIX for t in texts]
    return processor(text=wrapped, return_tensors="pt", padding="longest")


def process_doc_images(processor, images):
    """Wrap document images in chat template with representation instruction.

    Returns batch-major pixel_values (B, max_patches, dim) so that
    chunk_inputs / GradCache can split along the batch dimension.
    BiQwen3.forward() un-pads back to flat before passing to Qwen3VLModel.
    """
    texts = [_DOC_IMAGE_TMPL] * len(images)
    batch = processor(text=texts, images=images, return_tensors="pt", padding="longest")

    # Reshape pixel_values from flat (total_patches, dim) to batch-major
    # (B, max_patches, dim) for GradCache chunking compatibility.
    if "pixel_values" in batch and "image_grid_thw" in batch:
        offsets = batch["image_grid_thw"].prod(dim=1).tolist()
        pixel_chunks = list(torch.split(batch["pixel_values"], offsets))
        batch["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_chunks, batch_first=True
        )

    return batch


def make_text_collate_fn(processor, num_hard_negatives=0):
    """Collate for text-only training: queries and doc texts both as text."""

    def collate(batch):
        queries = [item[0] for item in batch]
        doc_texts = []
        for item in batch:
            doc_texts.append(item[1])  # positive passage
            neg_texts = item[2] if len(item) > 2 else []
            for nt in neg_texts:
                doc_texts.append(nt)

        query_inputs = process_queries(processor, list(queries))
        doc_inputs = process_doc_texts(processor, doc_texts)
        return query_inputs, doc_inputs

    return collate


def make_collate_fn(processor, num_hard_negatives=0):
    """Collate: process queries as text, images as visual prompts.

    With hard negatives, document images are interleaved:
        [pos1, neg1a, neg1b, pos2, neg2a, neg2b, ...]
    so that document.size(0) = batch_size * (1 + num_hard_negatives).
    """

    def _load_image(path):
        with Image.open(path) as im:
            return im.convert("RGB")

    def collate(batch):
        t_start = time.time()
        queries = [item[0] for item in batch]

        # Build document image list: positive + hard negatives per query
        doc_images = []
        for item in batch:
            pos_path = item[1]
            doc_images.append(_load_image(pos_path))
            neg_paths = item[2] if len(item) > 2 else []
            for np_ in neg_paths:
                if np_ is not None:
                    doc_images.append(_load_image(np_))
                else:
                    doc_images.append(_load_image(pos_path))
        t_io = time.time()

        query_inputs = process_queries(processor, list(queries))
        t_q = time.time()
        image_inputs = process_doc_images(processor, doc_images)
        t_d = time.time()
        total = t_d - t_start
        if total > 5:
            logger.warning(
                f"Slow collate: io={t_io - t_start:.1f}s q={t_q - t_io:.1f}s "
                f"img={t_d - t_q:.1f}s total={total:.1f}s"
            )
        return query_inputs, image_inputs

    return collate


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _clear_rope_deltas(model):
    """Clear stale rope_deltas state on Qwen3VL model.

    Qwen3VLModel.compute_3d_position_ids stores self.rope_deltas after processing
    image batches. On subsequent text-only forwards, it reuses this stale state,
    causing shape mismatch: position_ids (3, batch, seq_len) + delta (old_batch, 1).
    Must clear between image→text forward transitions.
    """
    inner = model
    # Unwrap PeftModel → BiQwen3 → Qwen3VLModel
    while hasattr(inner, "model"):
        inner = inner.model
    if hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None


def forward_query(model, inputs, bidirectional=False):
    """Forward text query through model → normalized embedding."""
    _clear_rope_deltas(model)
    return model(**inputs, bidirectional=bidirectional)


def forward_doc(model, inputs, bidirectional=False):
    """Forward image document through model → normalized embedding."""
    _clear_rope_deltas(model)
    return model(**inputs, bidirectional=bidirectional)


def chunk_inputs(inputs, chunk_size):
    """Split a batch of inputs into chunks along batch dimension.

    Only includes tensor values in chunks (non-tensors are dropped) so that
    RandContext's get_device_states doesn't crash on non-tensor dict values.
    """
    batch_size = next(
        v.shape[0] for v in inputs.values() if isinstance(v, torch.Tensor)
    )
    # Verify all tensors are batch-major (first dim == batch_size).
    # pixel_values must be (B, max_patches, dim), not flattened (sum_patches, dim).
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.shape[0] != batch_size:
            raise ValueError(
                f"chunk_inputs: {k}.shape[0]={v.shape[0]} != batch_size={batch_size}. "
                f"All tensors must be batch-major for chunking."
            )
    chunks = []
    for start in range(0, batch_size, chunk_size):
        chunk = {
            k: v[start : start + chunk_size]
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Retrieval evaluation (global QxM retrieval with expanded doc pool)
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(q_embs, i_embs, gold_doc_indices):
    """Compute retrieval metrics for a query set against a larger document pool."""
    sims = q_embs @ i_embs.T  # (Q, M)
    gold_doc_indices = np.asarray(gold_doc_indices)
    rankings = (-sims).argsort(axis=1)
    ranks = np.argmax(rankings == gold_doc_indices[:, None], axis=1)
    pos_sims = sims[np.arange(sims.shape[0]), gold_doc_indices]

    if sims.shape[1] > 1:
        neg_sum = sims.sum(axis=1) - pos_sims
        mean_neg_per_query = neg_sum / (sims.shape[1] - 1)
        mean_neg_sim = float(mean_neg_per_query.mean())
        margin = float((pos_sims - mean_neg_per_query).mean())
    else:
        mean_neg_sim = 0.0
        margin = float(pos_sims.mean())

    return {
        "recall@1": float((ranks < 1).mean()),
        "recall@5": float((ranks < 5).mean()),
        "recall@10": float((ranks < 10).mean()),
        "mrr": float((1.0 / (ranks + 1)).mean()),
        "mean_pos_sim": float(pos_sims.mean()),
        "mean_neg_sim": mean_neg_sim,
        "margin": margin,
    }


def resolve_jsonl_path(jsonl_path, path_str):
    """Resolve a possibly relative path against a JSONL file location."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path.resolve())
    return str((Path(jsonl_path).resolve().parent / path).resolve())


def wikipedia_url_to_slug(url):
    """Normalize an enwiki URL to an article slug."""
    if not url or "/wiki/" not in url:
        return None
    slug = unquote(url.split("/wiki/")[-1]).replace(" ", "_").split("#")[0]
    if not slug or slug.startswith("Category:"):
        return None
    return slug


def load_slug_to_article_id(articles_json):
    """Load articles.json and build slug -> article_id map."""
    with open(articles_json) as f:
        articles = json.load(f)
    return {slug: idx for idx, slug in enumerate(articles)}


def load_retrieval_queries(jsonl_path, max_examples=0):
    """Load query -> gold image pairs for retrieval eval against the full datastore."""
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            chunk_path = resolve_jsonl_path(jsonl_path, item["chunk_path"])
            examples.append(
                {
                    "query": item["query"],
                    "gold_path": chunk_path,
                }
            )
            if max_examples > 0 and len(examples) >= max_examples:
                break
    logger.info(f"Loaded {len(examples)} retrieval queries from {jsonl_path}")
    return examples


def load_simpleqa_queryset(jsonl_path=None, max_examples=1000, articles_json=None):
    """Load SimpleQA queryset JSONL bundled with this repo."""
    examples = []
    slug_to_aid = None
    if jsonl_path and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                gold_article_ids = list(item.get("gold_article_ids", []))
                if not gold_article_ids and articles_json:
                    if slug_to_aid is None:
                        slug_to_aid = load_slug_to_article_id(articles_json)
                    seen = set()
                    for url in item.get("urls", []):
                        slug = wikipedia_url_to_slug(url)
                        aid = slug_to_aid.get(slug) if slug else None
                        if aid is not None and aid not in seen:
                            seen.add(aid)
                            gold_article_ids.append(aid)
                examples.append(
                    {
                        "id": item.get("id", str(len(examples))),
                        "query": item.get("query", item.get("problem", "")),
                        "answer": item.get("answer", ""),
                        "urls": item.get("urls", []),
                        "gold_article_ids": gold_article_ids,
                    }
                )
                if max_examples > 0 and len(examples) >= max_examples:
                    break
        logger.info(f"Loaded {len(examples)} SimpleQA queries from {jsonl_path}")
        return examples

    raise FileNotFoundError(f"SimpleQA queryset not found at {jsonl_path!r}.")


@torch.no_grad()
def embed_query_texts(model, processor, queries, device, batch_size=128):
    """Embed text queries with the current query tower."""
    raw = model.module if hasattr(model, "module") else model
    all_embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        inputs = process_queries(processor, batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _clear_rope_deltas(raw)
            emb = raw(**inputs, bidirectional=getattr(raw, "_bidirectional", False))
        all_embs.append(emb.cpu().float().numpy())
    if not all_embs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(all_embs, axis=0)


def fetch_tile_image(search_api_url, path, timeout=30, retries=3):
    """Fetch a tile image via the search API's /tile endpoint.

    Falls back to local file if the path exists locally.
    Retries on network errors with exponential backoff.
    """
    if os.path.exists(path):
        return Image.open(path)
    tile_url = (
        search_api_url.rstrip("/") + "/tile?" + urllib.parse.urlencode({"path": path})
    )
    for attempt in range(retries):
        try:
            req = urlrequest.Request(tile_url, method="GET")
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                return Image.open(io.BytesIO(resp.read()))
        except (TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = 2**attempt
                logger.warning(
                    f"fetch_tile_image attempt {attempt + 1}/{retries} failed for {path}: {e}, retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                raise


def search_api_by_embeddings(search_api_url, query_embeddings, n_docs=3, timeout=120):
    """Search the wiki-screenshot index using pre-computed query embeddings."""
    if query_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D query embeddings, got shape={query_embeddings.shape}"
        )
    payload = {
        "queries": [{"embedding": emb.tolist()} for emb in query_embeddings],
        "n_docs": n_docs,
    }
    req = urlrequest.Request(
        search_api_url.rstrip("/") + "/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Search API HTTP {e.code}: {body}") from e
    except urlerror.URLError as e:
        raise RuntimeError(f"Search API request failed: {e}") from e


@torch.no_grad()
def run_search_api_retrieval_eval(
    model, processor, examples, device, search_api_url, batch_size=32, n_docs=3
):
    """Compute exact-image Recall@1/3 against the full search datastore."""
    was_training = model.training
    model.eval()

    def _normalize_tile_path(path_str):
        """Normalize tile path for matching.

        Absolute paths are used as-is. Relative paths starting with 'images/'
        (from HF dataset) are matched by shard suffix for backward compatibility
        with search APIs that return absolute paths.
        """
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        # Relative path (e.g. images/shard_XXX/...) — extract shard suffix
        parts = p.parts
        for i, part in enumerate(parts):
            if part.startswith("shard_"):
                return "/".join(parts[i:])
        return path_str

    def _match_paths(gold, hit):
        """Check if gold and hit refer to the same tile."""
        if gold == hit:
            return True
        # Fallback: compare shard suffixes
        g_parts = Path(gold).parts
        h_parts = Path(hit).parts
        for gi, gp in enumerate(g_parts):
            if gp.startswith("shard_"):
                g_suffix = "/".join(g_parts[gi:])
                for hi, hp in enumerate(h_parts):
                    if hp.startswith("shard_"):
                        return g_suffix == "/".join(h_parts[hi:])
        return False

    recall1 = 0
    recall3 = 0
    total = 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        queries = [item["query"] for item in batch]
        gold_paths = [_normalize_tile_path(item["gold_path"]) for item in batch]
        query_embs = embed_query_texts(
            model, processor, queries, device, batch_size=batch_size
        )
        search_resp = search_api_by_embeddings(
            search_api_url, query_embs, n_docs=n_docs
        )
        for gold_path, result in zip(gold_paths, search_resp["results"]):
            hit_paths = [hit["path"] for hit in result["hits"]]
            total += 1
            if hit_paths and _match_paths(gold_path, hit_paths[0]):
                recall1 += 1
            if any(_match_paths(gold_path, hp) for hp in hit_paths[:3]):
                recall3 += 1

    if was_training:
        model.train()
    if total == 0:
        return {"recall@1": 0.0, "recall@3": 0.0}
    return {
        "recall@1": recall1 / total,
        "recall@3": recall3 / total,
    }


def run_simpleqa_search_api_eval(
    model,
    processor,
    examples,
    device,
    search_api_url,
    vllm_url="",
    vllm_model="",
    batch_size=32,
    n_docs=3,
    grader_model="gpt-4.1-2025-04-14",
    vllm_max_tokens=200,
    vllm_enable_thinking=False,
):
    """Run SimpleQA retrieval, compute article recall, then judge QA correctness."""
    import base64
    import io
    import re

    metrics = {}

    queries = [item["query"] for item in examples]
    query_embs = embed_query_texts(
        model, processor, queries, device, batch_size=batch_size
    )
    search_resp = search_api_by_embeddings(search_api_url, query_embs, n_docs=n_docs)

    if not vllm_url:
        logger.info("SimpleQA will use hosted OpenAI API (no --vllm-url provided)")

    # VQA answer client — uses vLLM if provided, else OpenAI
    # vLLM with multi-image VQA can take >60s per request
    answer_client = build_openai_client(vllm_url, timeout=180)
    answer_client.models.list()

    # Grader client — always uses hosted OpenAI API so we can use GPT-4.1
    grader_client = build_openai_client("", timeout=60)

    recall1 = 0
    recall3 = 0
    recall_total = 0
    correct = 0
    total = 0
    # --- Phase 1: compute recall from search results (URL-based, matching pixelrag eval) ---
    from urllib.parse import unquote

    def _norm_url(u):
        return unquote(u.strip().split("#")[0])

    def _find_wikipedia_url(urls):
        """Extract the first en.wikipedia.org URL from a list."""
        all_parts = []
        for raw in urls:
            for part in raw.split("\n"):
                part = part.strip().lstrip("- ").strip().split("#")[0]
                if "wikipedia.org/wiki/" in part:
                    all_parts.append(part)
        for part in all_parts:
            if "en.wikipedia.org/wiki/" in part:
                return part
        return (
            all_parts[0]
            if all_parts
            else (urls[0].split("#")[0].lstrip("- ").strip() if urls else None)
        )

    for example, result in zip(examples, search_resp["results"]):
        gt_url = _find_wikipedia_url(example.get("urls", []))
        if not gt_url:
            continue
        gt_url = _norm_url(gt_url)
        hit_urls = [_norm_url(hit.get("url", "")) for hit in result["hits"][:n_docs]]
        recall_total += 1
        if hit_urls and hit_urls[0] == gt_url:
            recall1 += 1
        if any(u == gt_url for u in hit_urls):
            recall3 += 1

    # --- Phase 2: VQA answers via vLLM (concurrent) ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _do_vqa(idx, example, result):
        try:
            # Query text FIRST, then images (matches naive baseline order)
            content_parts = [{"type": "text", "text": example["query"]}]
            for hit in result["hits"][:n_docs]:
                img = fetch_tile_image(search_api_url, hit["path"])
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            resp = answer_client.chat.completions.create(
                model=vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant who answers questions based on provided evidence.\nUse <think></think> tags to show your reasoning if needed.\nAnswer the question directly and concisely based ONLY on the provided evidence.",
                    },
                    {"role": "user", "content": content_parts},
                ],
                max_tokens=vllm_max_tokens,
                temperature=0,
                **(
                    {
                        "extra_body": {
                            "chat_template_kwargs": {
                                "enable_thinking": vllm_enable_thinking
                            }
                        }
                    }
                    if "Qwen3.5" in vllm_model or vllm_enable_thinking
                    else {}
                ),
            )
            predicted = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"SimpleQA [{idx + 1}] VQA failed: {e}")
            predicted = ""
        return idx, predicted

    # vLLM concurrency limited to avoid OOM on the single GPU
    vqa_concurrency = 4
    predictions = [""] * len(examples)
    logger.info(
        f"SimpleQA: sending {len(examples)} VQA requests to vLLM (concurrency={vqa_concurrency})"
    )
    with ThreadPoolExecutor(max_workers=vqa_concurrency) as pool:
        futures = [
            pool.submit(_do_vqa, i, ex, res)
            for i, (ex, res) in enumerate(zip(examples, search_resp["results"]))
        ]
        for fut in as_completed(futures):
            idx, pred = fut.result()
            predictions[idx] = pred
            if (idx + 1) % 20 == 0:
                logger.info(f"SimpleQA VQA: {idx + 1}/{len(examples)} done")
    logger.info(f"SimpleQA VQA: all {len(examples)} done")

    # --- Phase 3: grade with OpenAI (concurrent) ---
    def _do_grade(idx, example, predicted):
        try:
            grade_resp = grader_client.chat.completions.create(
                model=grader_model,
                messages=[
                    {
                        "role": "user",
                        "content": _GRADER_TEMPLATE.format(
                            question=example["query"],
                            target=example["answer"],
                            predicted_answer=predicted,
                        ),
                    }
                ],
                max_tokens=5,
                temperature=0,
            )
            grade = grade_resp.choices[0].message.content.strip()
            return idx, bool(re.search(r"A", grade))
        except Exception as e:
            logger.warning(f"SimpleQA [{idx + 1}] grading failed: {e}")
            return idx, False

    gradeable = [
        (i, ex, predictions[i]) for i, ex in enumerate(examples) if ex.get("answer")
    ]
    total = len(gradeable)
    if total > 0:
        logger.info(f"SimpleQA: grading {total} answers via OpenAI ({grader_model})")
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_do_grade, i, ex, pred) for i, ex, pred in gradeable]
            for fut in as_completed(futures):
                idx, is_correct = fut.result()
                if is_correct:
                    correct += 1

    if total > 0:
        metrics["qa_score"] = correct / total
        metrics["qa_correct"] = correct
        metrics["qa_total"] = total
    if recall_total > 0:
        metrics["recall@1"] = recall1 / recall_total
        metrics["recall@3"] = recall3 / recall_total
        metrics["recall_total"] = recall_total
    return metrics


_GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
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


@torch.no_grad()
def run_miniv6_eval(
    model,
    processor,
    test_data,
    device,
    batch_size=64,
    vllm_url="",
    vllm_model="",
    grader_model="gpt-4.1-2025-04-14",
    output_path=None,
    vllm_max_tokens=200,
    vllm_enable_thinking=False,
):
    """Evaluate on mini-v6 tiles: R@1, R@3, and optional QA score via vLLM.

    Args:
        output_path: If set, save per-example results as JSONL.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import re

    raw = model.module if hasattr(model, "module") else model
    raw.eval()

    questions = test_data["questions"]
    doc_paths = test_data["doc_paths"]
    golden_mapping = test_data["golden_mapping"]

    def _load_image(path):
        with Image.open(path) as im:
            return im.convert("RGB")

    # Embed queries
    t_eval_start = time.time()
    query_texts = [q["problem"] for q in questions]
    q_embs = []
    for i in range(0, len(query_texts), batch_size * 2):
        batch = query_texts[i : i + batch_size * 2]
        inputs = process_queries(processor, batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _clear_rope_deltas(raw)
            _bidir = getattr(raw, "_bidirectional", False)
            emb = raw(**inputs, bidirectional=_bidir)
        q_embs.append(emb.cpu().float().numpy())
    t_query_emb = time.time()
    logger.info(
        f"  [profile] query embed: {t_query_emb - t_eval_start:.1f}s ({len(query_texts)} queries)"
    )

    # Embed images — use cached preprocessed tensors if available
    max_px = processor.image_processor.max_pixels
    cache_path = os.path.join(
        os.path.dirname(doc_paths[0]),
        f".tile_cache_n{len(doc_paths)}_px{max_px}_bs{batch_size}.pt",
    )

    i_embs = []
    if os.path.exists(cache_path):
        # Fast path: load preprocessed batches from cache
        cached_batches = torch.load(cache_path, map_location="cpu", weights_only=True)
        logger.info(
            f"  [cache] loaded {len(cached_batches)} preprocessed tile batches from {cache_path}"
        )
        for inputs in cached_batches:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _clear_rope_deltas(raw)
                emb = raw(**inputs, bidirectional=_bidir)
            i_embs.append(emb.cpu().float().numpy())
    else:
        # Slow path: preprocess from images, then save cache
        cached_batches = []
        pool = ThreadPoolExecutor(max_workers=4)
        batched_paths = [
            doc_paths[i : i + batch_size] for i in range(0, len(doc_paths), batch_size)
        ]
        future = (
            pool.submit(
                lambda paths: list(pool.map(_load_image, paths)), batched_paths[0]
            )
            if batched_paths
            else None
        )
        for idx, _ in enumerate(batched_paths):
            images = future.result()
            if idx + 1 < len(batched_paths):
                next_paths = batched_paths[idx + 1]
                future = pool.submit(
                    lambda paths: list(pool.map(_load_image, paths)), next_paths
                )
            inputs = process_doc_images(processor, images)
            # Save CPU tensors for cache
            cached_batches.append({k: v.cpu() for k, v in inputs.items()})
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _clear_rope_deltas(raw)
                emb = raw(**inputs, bidirectional=_bidir)
            i_embs.append(emb.cpu().float().numpy())
            for img in images:
                img.close()
        pool.shutdown(wait=False)
        # Save cache for future evals
        try:
            torch.save(cached_batches, cache_path)
            logger.info(
                f"  [cache] saved {len(cached_batches)} preprocessed tile batches to {cache_path}"
            )
        except Exception as e:
            logger.warning(f"  [cache] failed to save: {e}")

    t_doc_emb = time.time()
    logger.info(
        f"  [profile] doc embed: {t_doc_emb - t_query_emb:.1f}s ({len(doc_paths)} tiles)"
    )

    q_embs = np.concatenate(q_embs, axis=0)
    i_embs = np.concatenate(i_embs, axis=0)

    # Compute R@1, R@3 using article-level matching + collect top-3 paths per query
    sims = q_embs @ i_embs.T  # (Q, D)
    r1 = r3 = has_golden = 0
    top3_per_query = []
    per_example_results = []
    for qi, q in enumerate(questions):
        top_idx = np.argsort(sims[qi])[::-1][:3]
        top3_paths = [doc_paths[i] for i in top_idx]
        top3_per_query.append(top3_paths)
        gids = golden_mapping.get(q["id"], [])
        rids = [
            os.path.basename(p).replace("dist_", "").split("_chunk_")[0]
            for p in top3_paths
        ]
        hit1 = bool(gids and rids[0] in gids)
        hit3 = bool(gids and any(rid in gids for rid in rids))
        if gids:
            has_golden += 1
            if hit1:
                r1 += 1
            if hit3:
                r3 += 1
        per_example_results.append(
            {
                "id": q["id"],
                "problem": q["problem"],
                "answer": q.get("answer", ""),
                "top3_paths": top3_paths,
                "top3_article_ids": rids,
                "golden_ids": gids,
                "hit@1": hit1,
                "hit@3": hit3,
            }
        )

    t_ranking = time.time()
    logger.info(f"  [profile] ranking: {t_ranking - t_doc_emb:.1f}s")

    metrics = {
        "recall@1": r1 / has_golden if has_golden else 0,
        "recall@3": r3 / has_golden if has_golden else 0,
    }

    # QA scoring: VQA via vLLM, grading via OpenAI (GPT-4.1)
    if vllm_url:
        try:
            import base64

            os.environ.get("VLLM_API_KEY", "dummy")
            answer_client = build_openai_client(vllm_url, timeout=180)
            grader_client = build_openai_client("", timeout=60)

            def _do_qa(qi):
                q = questions[qi]
                answer = q.get("answer", "")
                if not answer:
                    return qi, "", "", False
                # Query text FIRST, then images
                content_parts = [{"type": "text", "text": q["problem"]}]
                for p in top3_per_query[qi]:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                try:
                    resp = answer_client.chat.completions.create(
                        model=vllm_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a research assistant who answers questions based on provided evidence.\nUse <think></think> tags to show your reasoning if needed.\nAnswer the question directly and concisely based ONLY on the provided evidence.",
                            },
                            {"role": "user", "content": content_parts},
                        ],
                        max_tokens=vllm_max_tokens,
                        temperature=0,
                        **(
                            {
                                "extra_body": {
                                    "chat_template_kwargs": {
                                        "enable_thinking": vllm_enable_thinking
                                    }
                                }
                            }
                            if "Qwen3.5" in vllm_model or vllm_enable_thinking
                            else {}
                        ),
                    )
                    predicted = resp.choices[0].message.content.strip()
                except Exception:
                    predicted = ""
                # Grade with OpenAI GPT-4.1
                is_correct = False
                try:
                    grade_resp = grader_client.chat.completions.create(
                        model=grader_model,
                        messages=[
                            {
                                "role": "user",
                                "content": _GRADER_TEMPLATE.format(
                                    question=q["problem"],
                                    target=answer,
                                    predicted_answer=predicted,
                                ),
                            }
                        ],
                        max_tokens=5,
                        temperature=0,
                    )
                    grade = grade_resp.choices[0].message.content.strip()
                    if re.search(r"A", grade):
                        is_correct = True
                except Exception:
                    grade = ""
                return qi, predicted, grade, is_correct

            correct = total = 0
            with ThreadPoolExecutor(max_workers=4) as qa_pool:
                futures = [
                    qa_pool.submit(_do_qa, qi)
                    for qi in range(len(questions))
                    if questions[qi].get("answer")
                ]
                for fut in as_completed(futures):
                    qi, predicted, grade, is_correct = fut.result()
                    per_example_results[qi]["predicted"] = predicted
                    per_example_results[qi]["grade"] = grade
                    per_example_results[qi]["correct"] = is_correct
                    if is_correct:
                        correct += 1
                    total += 1

            if total > 0:
                metrics["qa_score"] = correct / total
                t_qa = time.time()
                logger.info(
                    f"  QA: {correct}/{total} = {correct / total:.3f} "
                    f"[profile: {t_qa - t_ranking:.1f}s]"
                )
        except Exception as e:
            logger.warning(f"QA eval failed: {e}")

    # Save per-example results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for item in per_example_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"  Saved {len(per_example_results)} eval results to {output_path}")

    logger.info(f"  [profile] eval total: {time.time() - t_eval_start:.1f}s")
    raw.train()
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument(
        "--mode",
        choices=["standard", "query-side-tune"],
        default="standard",
        help="standard = shared query/doc tower, query-side-tune = "
        "train LoRA only on query tower while freezing doc tower",
    )
    parser.add_argument(
        "--query-side-backward",
        choices=["gradcache", "direct"],
        default="direct",
        help="Backward path for query-side-tune: direct (default, more stable) or gradcache",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Emit detailed per-rank trace logs around suspicious sync points",
    )
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--train-jsonl", default="training/data/train.jsonl")
    parser.add_argument(
        "--data-split-dir",
        default=None,
        help=(
            "Directory containing train_hn.jsonl, eval_hn.jsonl, and test_hn.jsonl. "
            "If set, overrides --train-jsonl/--eval-jsonl."
        ),
    )
    parser.add_argument(
        "--img-mix",
        nargs="+",
        metavar="SPEC",
        default=None,
        help=(
            "Mixed-source image training. Each SPEC is name:jsonl_path:count. "
            "Counts must sum to --batch-size. "
            "Example: wiki:data/wiki.jsonl:44 moca:data/moca.jsonl:5"
        ),
    )
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        default=True,
        help="Skip Image.open().verify() during data init (default: True, data is pre-validated)",
    )
    parser.add_argument("--eval-jsonl", default="training/data/eval.jsonl")
    parser.add_argument(
        "--test-jsonl",
        default="training/data/test.jsonl",
        help="Held-out retrieval test split used by query-side-tune mode",
    )
    parser.add_argument(
        "--test-data",
        nargs="+",
        default=["training/data/test_miniv6.json"],
        help="One or more test JSONs (each with questions, golden_mapping, tiles_dir). "
        "Each set is evaluated separately; metrics logged as test_<name>/* "
        "where <name> is derived from the filename (e.g. test_miniv6.json -> miniv6).",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Batch size for mini-v6/v7 test eval (lower to avoid OOM)",
    )
    parser.add_argument(
        "--test-eval-steps",
        type=int,
        default=250,
        help="Run mini-v6 retrieval eval every N steps (0=disable)",
    )
    parser.add_argument(
        "--search-api-url",
        default="http://localhost:30888",
        help="wiki-screenshot search API base URL for query-side-tune evals",
    )
    parser.add_argument(
        "--articles-json",
        default="/opt/dlami/nvme/kiwix/wikipedia_en_all_maxi_2025-08.zim.articles.json",
        help="Wikipedia articles.json used to map SimpleQA URLs to article ids",
    )
    parser.add_argument(
        "--search-api-batch-size",
        type=int,
        default=32,
        help="Batch size for local query embedding before hitting search API",
    )
    parser.add_argument(
        "--simpleqa-jsonl",
        default="training/data/simpleqa_wiki_1k_queryset.jsonl",
        help="Bundled SimpleQA queryset JSONL",
    )
    parser.add_argument(
        "--simpleqa-max-examples",
        type=int,
        default=1000,
        help="Number of SimpleQA examples to evaluate in query-side-tune mode",
    )
    parser.add_argument(
        "--simpleqa-grader-model",
        default="gpt-4.1-2025-04-14",
        help="LLM-as-judge model used by PixelRAG SimpleQA evaluation",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8201/v1",
        help="vLLM OpenAI-compatible URL for QA grading (empty to disable)",
    )
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--vllm-enable-thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3.5/Qwen3 VLLM reader",
    )
    parser.add_argument(
        "--vllm-max-tokens",
        type=int,
        default=200,
        help="Max tokens for VLLM reader response",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; just load model (and --resume ckpt) and run mini-v6 eval, then exit",
    )
    parser.add_argument("--output-dir", default="training/output_contrastors")
    parser.add_argument("--resume", type=str, default=None)
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count per GPU process",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor per worker",
    )
    parser.add_argument(
        "--grad-cache-chunk",
        type=int,
        default=2,
        help="GradCache chunk size (forward this many at a time)",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Warmup steps (default: 5% of max_steps)",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "constant", "cosine-restarts"],
        default="constant",
        help="LR scheduler: 'constant', 'cosine' (decay to 0), or 'cosine-restarts' (periodic restarts)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=2,
        help="Number of cosine cycles for cosine-restarts scheduler",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Initial temperature for learnable logit scale (1/temp = init scale)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (default 0.01)",
    )
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=20,
        help="Max eval batches to run (0=all)",
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=0,
        help="Number of hard negatives per query (requires neg_chunk_paths in JSONL)",
    )
    parser.add_argument(
        "--in-batch-only",
        action="store_true",
        help="Ablation flag: force in-batch negatives only (forces --num-hard-negatives=0). "
        "Overrides --num-hard-negatives if both are passed.",
    )
    parser.add_argument(
        "--hardness-alpha",
        type=float,
        default=0.0,
        help="LLaVE hardness weighting alpha (0=off). Upweights harder negatives in softmax. Try 5-9.",
    )
    # Text warmup
    parser.add_argument(
        "--text-warmup-steps",
        type=int,
        default=0,
        help="Number of text-only warmup steps before image training (0=disabled)",
    )
    parser.add_argument(
        "--text-data-dir",
        type=str,
        default=None,
        help="Directory containing text-qa-pair JSONL files (chunk_*/filtered_hn.jsonl)",
    )
    parser.add_argument(
        "--text-mix-ratio",
        type=float,
        default=0.0,
        help="Fraction of steps that use text batches during image phase (0=none, 0.2=every 5th step)",
    )
    parser.add_argument(
        "--text-curriculum",
        action="store_true",
        help="Gradually decrease text ratio: 50%% → 33%% → 20%% → 0%% over 4 equal phases",
    )
    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--no-lora-vit",
        action="store_false",
        dest="lora_vit",
        help="Disable LoRA on ViT (use LLM-only LoRA)",
    )
    parser.add_argument(
        "--lora-vit",
        action="store_true",
        default=True,
        help="Also apply LoRA to ViT vision encoder (attn + MLP) and merger. On by default.",
    )
    parser.add_argument(
        "--lora-mlp",
        action="store_true",
        help="Also apply LoRA to LLM MLP layers (down_proj, gate_proj, up_proj)",
    )
    parser.add_argument(
        "--lora-vit-r",
        type=int,
        default=None,
        help="Separate LoRA rank for ViT layers (default: same as --lora-r)",
    )
    parser.add_argument(
        "--unfreeze-vit",
        action="store_true",
        help="Fully unfreeze ViT (no LoRA), only apply LoRA to LLM",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional attention instead of causal for embedding (like Nemotron ColEmbed V2)",
    )
    parser.add_argument(
        "--two-stage-lr",
        action="store_true",
        help="Reset LR scheduler after text warmup (independent cosine for text and image phases)",
    )
    parser.add_argument(
        "--lora-vit-attn-only",
        action="store_true",
        help="Only apply ViT LoRA to attention (qkv, proj), skip MLP layers",
    )
    parser.add_argument(
        "--reverse-data",
        action="store_true",
        help="Reverse training data order (later=higher quality data seen first) and disable shuffle",
    )
    parser.add_argument(
        "--dora",
        action="store_true",
        help="Use DoRA (Weight-Decomposed LoRA) instead of standard LoRA",
    )
    parser.add_argument(
        "--rslora",
        action="store_true",
        help="Use rsLoRA (Rank-Stabilized LoRA, alpha/sqrt(r) scaling) — useful for higher LoRA ranks",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default 0.05)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Use EMA (Exponential Moving Average) of model params for eval",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay rate (default 0.999)"
    )
    parser.add_argument(
        "--loss",
        choices=["infonce", "siglip"],
        default="infonce",
        help="Loss function: 'infonce' (standard softmax CE) or 'siglip' (pairwise sigmoid)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for InfoNCE loss (0=off, try 0.05-0.1 for regularization)",
    )
    # Resolution
    parser.add_argument("--max-num-visual-tokens", type=int, default=4096)
    # Wandb
    parser.add_argument("--wandb-project", default="wiki-screenshot-training")
    parser.add_argument(
        "--wandb-run-name", default=None, help="Run name (auto-generated if not set)"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    if args.data_split_dir:
        split_dir = Path(args.data_split_dir)
        train_candidates = [split_dir / "train_hn.jsonl", split_dir / "train.jsonl"]
        eval_candidates = [split_dir / "eval_hn.jsonl", split_dir / "eval.jsonl"]
        test_candidates = [split_dir / "test_hn.jsonl", split_dir / "test.jsonl"]
        for candidate in train_candidates:
            if candidate.exists():
                args.train_jsonl = str(candidate)
                break
        else:
            args.train_jsonl = str(train_candidates[0])
        for candidate in eval_candidates:
            if candidate.exists():
                args.eval_jsonl = str(candidate)
                break
        else:
            args.eval_jsonl = str(eval_candidates[0])
        for candidate in test_candidates:
            if candidate.exists():
                args.test_jsonl = str(candidate)
                break
        else:
            args.test_jsonl = str(test_candidates[0])
    if args.warmup_steps is None:
        args.warmup_steps = max(1, (args.max_steps + 19) // 20)
    if args.in_batch_only and args.num_hard_negatives != 0:
        print(
            f"[--in-batch-only] forcing --num-hard-negatives 0 "
            f"(was {args.num_hard_negatives})"
        )
        args.num_hard_negatives = 0

    # Distributed setup
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")
        rank = 0
        world_size = 1

    is_main = rank == 0

    simpleqa_api_ready = True
    if args.mode == "query-side-tune" and args.simpleqa_max_examples > 0:
        if is_main:
            simpleqa_api_ready = preflight_simpleqa_client(
                vllm_url=args.vllm_url,
                model_name=args.vllm_model,
            )
            # Also check that OPENAI_API_KEY is set for the grader
            if simpleqa_api_ready and not os.environ.get("OPENAI_API_KEY"):
                logger.warning(
                    "OPENAI_API_KEY not set — SimpleQA grading (judge) will be disabled."
                )
                simpleqa_api_ready = False
        if distributed:
            ready_tensor = torch.tensor([1 if simpleqa_api_ready else 0], device=device)
            dist.broadcast(ready_tensor, src=0)
            simpleqa_api_ready = bool(ready_tensor.item())
        if not simpleqa_api_ready:
            args.simpleqa_max_examples = 0
            if is_main:
                logger.warning(
                    "Disabling SimpleQA eval for this run because no working OpenAI / "
                    "OpenAI-compatible API was available at startup."
                )

    # Wandb init (rank 0 only)
    use_wandb = is_main and not args.no_wandb
    if use_wandb:
        import wandb

        wandb_config = vars(args).copy()
        wandb_config["world_size"] = world_size
        wandb_config["distributed"] = distributed
        # Auto-generate run name from key params if not specified
        run_name = args.wandb_run_name
        if run_name is None:
            data_name = Path(args.train_jsonl).stem
            mode_tag = f"{args.mode}_" if args.mode != "standard" else ""
            run_name = (
                f"{mode_tag}{data_name}_lr{args.lr}_bs{args.batch_size}x{world_size}"
                f"_hn{args.num_hard_negatives}_wu{args.warmup_steps}_steps{args.max_steps}"
            )
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            tags=[
                f"{world_size}gpu",
                f"lora-r{args.lora_r}",
                f"vt{args.max_num_visual_tokens}",
            ],
        )

    if is_main:
        logger.info(
            f"Training with {world_size} GPU(s), GradCache chunk={args.grad_cache_chunk}"
        )
        if args.mode == "query-side-tune" and args.query_side_backward == "gradcache":
            logger.warning(
                "query-side-tune with GradCache is experimental; direct mode is the stable default"
            )

    # Model
    from models.biqwen3 import BiQwen3
    from transformers import AutoProcessor

    model = BiQwen3.from_pretrained(args.model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    # Convert max_num_visual_tokens to pixel limits for Qwen3-VL processor.
    # Each visual token covers (patch_size * spatial_merge_size)^2 = 28^2 = 784 pixels.
    pixels_per_token = (model.patch_size * model.spatial_merge_size) ** 2
    processor.image_processor.max_pixels = args.max_num_visual_tokens * pixels_per_token
    processor.image_processor.min_pixels = max(
        processor.image_processor.min_pixels, 4 * pixels_per_token
    )
    processor.image_processor.size["longest_edge"] = (
        processor.image_processor.max_pixels
    )
    processor.image_processor.size["shortest_edge"] = (
        processor.image_processor.min_pixels
    )
    if is_main:
        logger.info(
            f"Visual tokens: max={args.max_num_visual_tokens} "
            f"→ max_pixels={processor.image_processor.max_pixels}"
        )
    init_chat_templates(processor)

    doc_model = None
    # LoRA
    # LLM attention: q_proj, k_proj, v_proj, o_proj
    # ViT attention: attn.qkv (fused), attn.proj
    # ViT MLP: mlp.linear_fc1, mlp.linear_fc2
    # Merger: visual.merger.linear_fc1, visual.merger.linear_fc2
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_mlp:
        # Add LLM MLP layers (colpali style)
        lora_targets += ["down_proj", "gate_proj", "up_proj"]
        if is_main:
            logger.info("LoRA targets include LLM MLP layers")

    # Determine ViT LoRA rank (may differ from LLM rank)
    vit_r = args.lora_vit_r if args.lora_vit_r is not None else args.lora_r
    dora_kwargs = {"use_dora": True} if getattr(args, "dora", False) else {}
    rslora_kwargs = {"use_rslora": True} if getattr(args, "rslora", False) else {}
    extra_lora_kwargs = {**dora_kwargs, **rslora_kwargs}
    if getattr(args, "dora", False) and rank == 0:
        logger.info("Using DoRA (Weight-Decomposed LoRA)")
    if getattr(args, "rslora", False) and rank == 0:
        logger.info("Using rsLoRA (Rank-Stabilized LoRA, alpha/sqrt(r) scaling)")

    if args.unfreeze_vit:
        # Full finetune ViT: apply LoRA to LLM only, then unfreeze ViT
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_targets,
            lora_dropout=args.lora_dropout,
            task_type="FEATURE_EXTRACTION",
            **extra_lora_kwargs,
        )
        model = get_peft_model(model, lora_config)
        # Unfreeze all ViT parameters
        vit_unfrozen = 0
        for name, param in model.named_parameters():
            if "visual." in name and "lora_" not in name:
                param.requires_grad = True
                vit_unfrozen += 1
        if is_main:
            logger.info(f"ViT fully unfrozen: {vit_unfrozen} params set to trainable")
    elif args.lora_vit:
        # Add ViT layers
        if args.lora_vit_attn_only:
            vit_targets = ["attn.qkv", "attn.proj"]
            if is_main:
                logger.info("ViT LoRA: attention only (no MLP)")
        else:
            vit_targets = ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]
        if vit_r != args.lora_r:
            # Single LoRA config with rank_pattern for per-module rank override
            all_targets = lora_targets + vit_targets
            rank_pat = {vt: vit_r for vt in vit_targets}
            alpha_pat = {vt: vit_r for vt in vit_targets}
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=all_targets,
                rank_pattern=rank_pat,
                alpha_pattern=alpha_pat,
                lora_dropout=args.lora_dropout,
                task_type="FEATURE_EXTRACTION",
                **extra_lora_kwargs,
            )
            model = get_peft_model(model, lora_config)
            if is_main:
                logger.info(
                    f"LoRA targets include ViT + merger layers "
                    f"(LLM r={args.lora_r}, ViT r={vit_r})"
                )
        else:
            lora_targets += vit_targets
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=lora_targets,
                lora_dropout=args.lora_dropout,
                task_type="FEATURE_EXTRACTION",
                **extra_lora_kwargs,
            )
            model = get_peft_model(model, lora_config)
            if is_main:
                logger.info("LoRA targets include ViT + merger layers")
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_targets,
            lora_dropout=args.lora_dropout,
            task_type="FEATURE_EXTRACTION",
            **extra_lora_kwargs,
        )
        model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model = model.to(device)
    # Set bidirectional flag on the underlying BiQwen3 model for eval functions
    raw_for_flag = model.module if hasattr(model, "module") else model
    if hasattr(raw_for_flag, "base_model"):  # PEFT wrapped
        raw_for_flag = raw_for_flag.base_model.model
    raw_for_flag._bidirectional = getattr(args, "bidirectional", False)
    if args.bidirectional and is_main:
        logger.info("Bidirectional attention enabled for embedding")

    # EMA initialization
    ema = None
    if getattr(args, "ema", False):
        ema = EMAModel(model, decay=getattr(args, "ema_decay", 0.999))
        if is_main:
            logger.info(f"EMA enabled (decay={args.ema_decay})")

    if args.mode == "query-side-tune":
        doc_model = BiQwen3.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
        doc_model.eval()
        doc_model.requires_grad_(False)
        if is_main:
            logger.info(
                "Query-side tune enabled: trainable query tower + frozen base doc tower"
            )

    # Learnable logit scale (contrastors/OpenCLIP pattern)
    logit_scale = LogitScale(init_value=1.0 / args.temperature).to(device)

    # Loss function selection
    if getattr(args, "loss", "infonce") == "siglip":
        active_loss_fn = siglip_loss
    elif getattr(args, "label_smoothing", 0.0) > 0:
        active_loss_fn = functools.partial(
            clip_loss, label_smoothing=args.label_smoothing
        )
    else:
        active_loss_fn = clip_loss
    if rank == 0:
        loss_desc = "pairwise sigmoid" if args.loss == "siglip" else "softmax InfoNCE"
        if getattr(args, "label_smoothing", 0.0) > 0:
            loss_desc += f" + label_smoothing={args.label_smoothing}"
        logger.info(f"Loss function: {args.loss} ({loss_desc})")

    # DDP: gradient sync is handled manually after all GradCache backward calls.
    # All surrogate backward passes run under no_sync(), then we manually
    # all_reduce gradients. This avoids find_unused_parameters issues where
    # query (text-only) and doc (image) chunks use different parameter subsets,
    # which confuses DDP's reducer and causes intermittent NCCL deadlocks.
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # Log model info to wandb
    if use_wandb:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        wandb.config.update(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": trainable / total * 100,
            },
            allow_val_change=True,
        )

    # Data
    if args.img_mix:
        # Mixed-source image training: each batch draws exact counts per source
        mix_sources = []
        for spec in args.img_mix:
            parts = spec.rsplit(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid --img-mix spec '{spec}', expected name:jsonl_path:count"
                )
            name, jsonl_path, count = parts[0], parts[1], int(parts[2])
            mix_sources.append((name, jsonl_path, count))
        total_count = sum(c for _, _, c in mix_sources)
        if total_count != args.batch_size:
            raise ValueError(
                f"--img-mix counts sum to {total_count}, but --batch-size is {args.batch_size}"
            )

        sub_datasets = []
        source_ranges = []
        offset = 0
        for name, jsonl_path, count in mix_sources:
            ds = QueryImageDataset(
                jsonl_path,
                num_hard_negatives=args.num_hard_negatives,
                skip_image_verify=args.skip_image_verify,
            )
            sub_datasets.append(ds)
            source_ranges.append((offset, len(ds), count))
            offset += len(ds)
            if is_main:
                logger.info(f"img-mix source '{name}': {len(ds)} pairs, {count}/batch")
        train_dataset = ConcatDataset(sub_datasets)
        train_sampler = MixedBatchSampler(
            source_ranges,
            shuffle=True,
            seed=42,
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            collate_fn=make_collate_fn(
                processor, num_hard_negatives=args.num_hard_negatives
            ),
        )
    else:
        train_dataset = QueryImageDataset(
            args.train_jsonl,
            num_hard_negatives=args.num_hard_negatives,
            skip_image_verify=args.skip_image_verify,
            reverse=getattr(args, "reverse_data", False),
        )
        do_shuffle = not getattr(args, "reverse_data", False)
        train_sampler = (
            DistributedSampler(train_dataset, shuffle=do_shuffle)
            if distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(do_shuffle and train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            collate_fn=make_collate_fn(
                processor, num_hard_negatives=args.num_hard_negatives
            ),
            drop_last=True,
        )

    eval_dataset = QueryImageDataset(
        args.eval_jsonl,
        num_hard_negatives=args.num_hard_negatives,
        skip_image_verify=args.skip_image_verify,
    )

    eval_batch_size = min(args.batch_size, args.grad_cache_chunk)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        collate_fn=make_collate_fn(
            processor, num_hard_negatives=args.num_hard_negatives
        ),
        drop_last=True,
    )

    # Text warmup data
    text_loader = None
    text_sampler = None
    if args.text_warmup_steps > 0 and args.text_data_dir:
        text_data_dir = Path(args.text_data_dir)
        text_jsonl_files = sorted(text_data_dir.glob("*/filtered_hn.jsonl"))
        if not text_jsonl_files:
            text_jsonl_files = sorted(text_data_dir.glob("*.jsonl"))
        if text_jsonl_files:
            text_dataset = TextQueryDataset(
                text_jsonl_files, num_hard_negatives=args.num_hard_negatives
            )
            text_sampler = (
                DistributedSampler(text_dataset, shuffle=True) if distributed else None
            )
            text_loader = DataLoader(
                text_dataset,
                batch_size=args.batch_size,
                shuffle=(text_sampler is None),
                sampler=text_sampler,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True,
                collate_fn=make_text_collate_fn(
                    processor, num_hard_negatives=args.num_hard_negatives
                ),
                drop_last=True,
            )
            if is_main:
                logger.info(
                    f"Text warmup: {len(text_dataset)} pairs, "
                    f"{args.text_warmup_steps} steps"
                )
        else:
            logger.warning(
                f"No text JSONL files found in {text_data_dir}, skipping text warmup"
            )

    # Load test datasets: list of {name, questions, golden_mapping, doc_paths}.
    # Multiple test sets are evaluated independently each test_eval step.
    test_datasets = []
    test_split_queries = None
    simpleqa_queries = None
    if is_main:
        if args.mode == "query-side-tune":
            if os.path.exists(args.test_jsonl):
                test_split_queries = load_retrieval_queries(args.test_jsonl)
            else:
                logger.warning(
                    f"Query-side retrieval test skipped: missing {args.test_jsonl}"
                )
            if args.simpleqa_max_examples > 0:
                try:
                    simpleqa_queries = load_simpleqa_queryset(
                        args.simpleqa_jsonl,
                        max_examples=args.simpleqa_max_examples,
                    )
                except Exception as e:
                    logger.warning(f"SimpleQA queryset load failed: {e}")
        elif args.test_eval_steps > 0:
            for tpath in args.test_data:
                if not os.path.exists(tpath):
                    logger.warning(f"Test data missing, skipping: {tpath}")
                    continue
                with open(tpath) as f:
                    td = json.load(f)
                tiles_dir = td["tiles_dir"]
                doc_paths = sorted(
                    [
                        os.path.join(tiles_dir, fn)
                        for fn in os.listdir(tiles_dir)
                        if fn.endswith(".png")
                    ]
                )
                # name = filename stem with leading "test_" stripped (test_miniv6.json -> miniv6)
                stem = os.path.splitext(os.path.basename(tpath))[0]
                name = stem[5:] if stem.startswith("test_") else stem
                test_datasets.append(
                    {
                        "name": name,
                        "questions": td["questions"],
                        "golden_mapping": td["golden_mapping"],
                        "doc_paths": doc_paths,
                    }
                )
                logger.info(
                    f"Loaded test '{name}': {len(td['questions'])} questions, "
                    f"{len(doc_paths)} tiles"
                )

    if use_wandb:
        wandb_test_cfg = {
            f"test_{td['name']}_queries": len(td["questions"]) for td in test_datasets
        }
        wandb_test_cfg.update(
            {f"test_{td['name']}_tiles": len(td["doc_paths"]) for td in test_datasets}
        )
        wandb.config.update(
            {
                "train_pairs": len(train_dataset),
                "eval_pairs": len(eval_dataset),
                **wandb_test_cfg,
                "test_queries": len(test_split_queries or []),
                "simpleqa_queries": len(simpleqa_queries or []),
                "effective_batch_size": args.batch_size * world_size,
            },
            allow_val_change=True,
        )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(logit_scale.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )
    elif args.scheduler == "cosine-restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
            num_cycles=args.num_cycles,
        )
    else:  # constant
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # Resume
    start_step = 0
    start_epoch = 0
    if args.resume:
        ts_path = Path(args.resume) / "training_state.pt"
        if ts_path.exists():
            ckpt = torch.load(ts_path, map_location=device)
            raw = model.module if distributed else model
            raw.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "logit_scale_state_dict" in ckpt:
                logit_scale.load_state_dict(ckpt["logit_scale_state_dict"])
            start_step = ckpt["step"]
            start_epoch = ckpt.get("epoch", 0)
            if is_main:
                logger.info(f"Resumed from step {start_step}, epoch {start_epoch}")
        else:
            # Adapter-only checkpoint (e.g. downloaded from HF) — load LoRA weights via safetensors
            adapter_path = Path(args.resume) / "adapter_model.safetensors"
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"Neither training_state.pt nor adapter_model.safetensors found in {args.resume}"
                )
            from safetensors.torch import load_file

            adapter_state = load_file(str(adapter_path))
            # Some PEFT versions save LoRA without the adapter-name suffix (".weight" vs ".default.weight")
            # Convert if needed by checking model's actual key format.
            raw = model.module if distributed else model
            model_keys = set(dict(raw.named_parameters()).keys())
            sample_lora_key = next((k for k in model_keys if "lora_A" in k), None)
            if sample_lora_key and ".default." in sample_lora_key:
                # Add ".default" before ".weight" if adapter_state lacks it
                converted = {}
                for k, v in adapter_state.items():
                    if (
                        k.endswith(".weight")
                        and ".default." not in k
                        and (
                            "lora_A" in k
                            or "lora_B" in k
                            or "lora_magnitude_vector" in k
                        )
                    ):
                        nk = k[: -len(".weight")] + ".default.weight"
                        converted[nk] = v
                    else:
                        converted[k] = v
                adapter_state = converted
            missing, unexpected = raw.load_state_dict(adapter_state, strict=False)
            # Filter out base-model "missing" keys (those aren't in adapter)
            real_missing = [
                k for k in missing if "lora_" in k or "modules_to_save" in k
            ]
            if is_main:
                logger.info(
                    f"Loaded adapter from {adapter_path}: {len(adapter_state)} tensors. "
                    f"Missing LoRA keys: {len(real_missing)}, unexpected: {len(unexpected)}"
                )
                if real_missing[:3]:
                    logger.warning(f"Sample missing LoRA: {real_missing[:3]}")
                if unexpected[:3]:
                    logger.warning(f"Sample unexpected: {unexpected[:3]}")
            # Try to infer step from path (e.g. .../checkpoint-150 or .../ckpt250)
            import re as _re

            m = _re.search(r"(?:checkpoint-|ckpt)(\d+)", str(args.resume))
            start_step = int(m.group(1)) if m else 0
            if is_main:
                logger.info(f"Adapter-only resume from inferred step {start_step}")

    # Graceful shutdown
    shutdown = False

    def handle_signal(*_):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Checkpoint helper
    def save_checkpoint(step, epoch):
        if not is_main:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw = model.module if distributed else model
        # Save LoRA adapter
        raw.save_pretrained(str(ckpt_dir))
        # Save training state
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "model_state_dict": raw.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "logit_scale_state_dict": logit_scale.state_dict(),
            },
            ckpt_dir / "training_state.pt",
        )
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    # Eval helper
    @torch.no_grad()
    def run_eval(step):
        if ema is not None:
            ema.apply(model)
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        raw = model.module if distributed else model
        raw_doc = doc_model if doc_model is not None else raw
        max_batches = (
            args.max_eval_batches if args.max_eval_batches > 0 else float("inf")
        )
        for batch in eval_loader:
            q_inputs, d_inputs = batch
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            d_inputs = {k: v.to(device) for k, v in d_inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _clear_rope_deltas(raw)
                q_emb = raw(**q_inputs)
                _clear_rope_deltas(raw_doc)
                d_emb = raw_doc(**d_inputs)
                loss, acc = active_loss_fn(
                    q_emb, d_emb, logit_scale, gather_enabled=False
                )
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
            if n_batches >= max_batches:
                break
        model.train()
        if n_batches > 0 and is_main:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            logger.info(
                f"eval step={step} loss={avg_loss:.4f} "
                f"acc={avg_acc:.4f} batches={n_batches}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "eval/loss": avg_loss,
                        "eval/accuracy": avg_acc,
                        "eval/batches": n_batches,
                    },
                    step=step,
                )

    def run_query_side_tests(step):
        if not is_main:
            return
        torch.cuda.empty_cache()
        if test_split_queries:
            try:
                logger.info(
                    f"Running test-split retrieval eval via search API: "
                    f"{len(test_split_queries)} queries..."
                )
                metrics = run_search_api_retrieval_eval(
                    model,
                    processor,
                    test_split_queries,
                    device,
                    search_api_url=args.search_api_url,
                    batch_size=args.search_api_batch_size,
                    n_docs=3,
                )
                for k, v in metrics.items():
                    logger.info(f"  test_split/{k}: {v:.4f}")
                if use_wandb:
                    wandb.log(
                        {f"test_split/{k}": v for k, v in metrics.items()}, step=step
                    )
            except Exception as e:
                logger.warning(f"test-split retrieval eval failed: {e}")

        if simpleqa_queries:
            try:
                logger.info(
                    f"Running SimpleQA eval via search API: {len(simpleqa_queries)} queries..."
                )
                metrics = run_simpleqa_search_api_eval(
                    model,
                    processor,
                    simpleqa_queries,
                    device,
                    search_api_url=args.search_api_url,
                    vllm_url=args.vllm_url,
                    vllm_model=args.vllm_model,
                    batch_size=args.search_api_batch_size,
                    n_docs=3,
                    grader_model=args.simpleqa_grader_model,
                    vllm_max_tokens=args.vllm_max_tokens,
                    vllm_enable_thinking=args.vllm_enable_thinking,
                )
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        logger.info(f"  simpleqa/{k}: {v:.4f}")
                    else:
                        logger.info(f"  simpleqa/{k}: {v}")
                if use_wandb and metrics:
                    wandb.log(
                        {
                            f"simpleqa/{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))
                        },
                        step=step,
                    )
            except Exception as e:
                logger.warning(f"SimpleQA eval failed: {e}")
        if ema is not None:
            ema.restore(model)
        model.train()

    # Train loop
    model.train()
    step = start_step
    epoch = start_epoch

    if is_main:
        logger.info(f"Starting training from step {start_step}")

    def run_test_evals(step, label_prefix="eval", log_to_wandb=True):
        """Run mini-v6/v8/... eval on every loaded test dataset and log per-set metrics."""
        if not test_datasets:
            return
        for td in test_datasets:
            name = td["name"]
            torch.cuda.empty_cache()
            logger.info(
                f"Running test '{name}': {len(td['questions'])} queries, "
                f"{len(td['doc_paths'])} tiles..."
            )
            metrics = run_miniv6_eval(
                model,
                processor,
                td,
                device,
                batch_size=args.test_batch_size,
                vllm_url=args.vllm_url,
                vllm_model=args.vllm_model,
                grader_model=args.simpleqa_grader_model,
                output_path=os.path.join(
                    args.output_dir, f"{label_prefix}_step{step}_{name}.jsonl"
                ),
                vllm_max_tokens=args.vllm_max_tokens,
                vllm_enable_thinking=args.vllm_enable_thinking,
            )
            for k, v in metrics.items():
                logger.info(f"  test_{name}/{k}: {v:.4f}")
            if log_to_wandb and use_wandb:
                wandb.log(
                    {f"test_{name}/{k}": v for k, v in metrics.items()}, step=step
                )

    # Eval-only mode: run all test evals at current step (post-resume) and exit
    if getattr(args, "eval_only", False) and is_main and test_datasets:
        suffix = ""
        if args.vllm_enable_thinking:
            suffix += "_think"
        suffix += f"_mt{args.vllm_max_tokens}"
        logger.info(
            f"[eval-only] Running test evals at step={start_step}, "
            f"thinking={args.vllm_enable_thinking}, max_tokens={args.vllm_max_tokens}"
        )
        run_test_evals(
            start_step, label_prefix=f"eval_only{suffix}", log_to_wandb=False
        )
        logger.info("[eval-only] Done. Exiting.")
        return

    # Step-0 baseline before any optimization updates.
    # For query-side-tune this gives retrieval recall / SimpleQA QA against the
    # frozen-base checkpoint, which is the most useful comparison point.
    if start_step == 0:
        if distributed:
            dist.barrier()
        if is_main and args.mode == "query-side-tune":
            logger.info("Running step-0 query-side baseline eval...")
            run_query_side_tests(step=0)
        elif is_main and test_datasets:
            logger.info("Running step-0 baseline test evals...")
            run_test_evals(0)
            model.train()
        if distributed:
            dist.barrier()

    def prefetched(loader):
        """Prefetch next batch in a background thread while GPU is busy."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            it = iter(loader)
            future = pool.submit(next, it, None)
            while True:
                batch = future.result()
                if batch is None:
                    break
                future = pool.submit(next, it, None)
                yield batch

    # Text warmup phase
    text_warmup_done = (
        args.text_warmup_steps <= 0
        or text_loader is None
        or step >= args.text_warmup_steps
    )
    if not text_warmup_done and is_main:
        logger.info(
            f"Starting text warmup phase: steps {step} → {args.text_warmup_steps}"
        )

    text_epoch = 0
    while not text_warmup_done and step < args.text_warmup_steps:
        if text_sampler:
            text_sampler.set_epoch(text_epoch)
        for batch in text_loader:
            if step >= args.text_warmup_steps:
                break
            if shutdown:
                break

            t0 = time.time()
            q_inputs, d_inputs = batch
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            d_inputs = {k: v.to(device) for k, v in d_inputs.items()}

            optimizer.zero_grad()
            q_chunks = chunk_inputs(q_inputs, args.grad_cache_chunk)
            d_chunks = chunk_inputs(d_inputs, args.grad_cache_chunk)

            loss, accuracy = grad_cache_loss(
                model=model,
                query_chunks=q_chunks,
                doc_chunks=d_chunks,
                logit_scale=logit_scale,
                query_process_fn=functools.partial(
                    forward_query, bidirectional=args.bidirectional
                ),
                doc_process_fn=functools.partial(
                    forward_doc, bidirectional=args.bidirectional
                ),
                hardness_alpha=args.hardness_alpha,
                loss_fn=active_loss_fn,
            )

            if args.max_grad_norm > 0:
                clip_grad_norm_(
                    list(model.parameters()) + list(logit_scale.parameters()),
                    args.max_grad_norm,
                )
            optimizer.step()
            logit_scale.clamp_()
            if ema is not None:
                ema.update(model)
            scheduler.step()
            step += 1
            dt = time.time() - t0

            if is_main:
                temp = 1.0 / logit_scale.log_scale.exp().item()
                cur_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[text-warmup] step={step}/{args.max_steps} "
                    f"loss={loss.item():.4f} acc={accuracy.item():.3f} "
                    f"lr={cur_lr:.2e} temp={temp:.4f} time={dt:.1f}s"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/accuracy": accuracy.item(),
                            "train/lr": cur_lr,
                            "train/temperature": temp,
                            "train/phase": 0,
                        },
                        step=step,
                    )

            need_eval = args.eval_steps > 0 and step % args.eval_steps == 0
            need_save = args.save_steps > 0 and step % args.save_steps == 0
            need_test = args.test_eval_steps > 0 and step % args.test_eval_steps == 0
            need_action = need_eval or need_test or need_save
            if need_action:
                if distributed:
                    dist.barrier()
                if need_eval and is_main:
                    torch.cuda.empty_cache()
                    run_eval(step)
                if need_save:
                    save_checkpoint(step, epoch)
                if need_test and is_main and test_datasets:
                    run_test_evals(step)
                    model.train()
                if distributed:
                    dist.barrier()
        text_epoch += 1

    if not text_warmup_done and is_main:
        logger.info(f"Text warmup complete at step {step}, switching to image training")
    text_warmup_done = True

    # Two-stage LR: reset scheduler for image phase with fresh cosine decay
    if args.two_stage_lr and args.text_warmup_steps > 0:
        image_steps = args.max_steps - args.text_warmup_steps
        if args.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=image_steps,
            )
        elif args.scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )
        # Reset lr to peak (optimizer stores current lr from text phase decay)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        if is_main:
            logger.info(
                f"Two-stage LR: reset scheduler for image phase "
                f"({image_steps} steps, warmup={args.warmup_steps})"
            )

    # Set up text interleaving for mixed training phase
    text_mix_iter = None
    text_mix_epoch = 0
    use_text_mix = (
        args.text_mix_ratio > 0 or args.text_curriculum
    ) and text_loader is not None

    def _get_text_mix_interval(current_step):
        """Return how often to insert a text batch (0 = never)."""
        if args.text_curriculum:
            # Curriculum: gradually decrease text ratio over image phase
            # Phase 1 (first 25%): 1:1 text:image (interval=2)
            # Phase 2 (25-50%):    1:2 (interval=3)
            # Phase 3 (50-75%):    1:4 (interval=5)
            # Phase 4 (last 25%):  no text (interval=0)
            image_steps = args.max_steps - args.text_warmup_steps
            progress = (current_step - args.text_warmup_steps) / max(image_steps, 1)
            if progress < 0.25:
                return 2  # 50% text
            elif progress < 0.50:
                return 3  # 33% text
            elif progress < 0.75:
                return 5  # 20% text
            else:
                return 0  # no text
        elif args.text_mix_ratio > 0:
            return max(1, round(1.0 / args.text_mix_ratio))
        return 0

    if use_text_mix and is_main:
        if args.text_curriculum:
            logger.info("Text curriculum enabled: 50%→33%→20%→0% over 4 phases")
        else:
            interval = _get_text_mix_interval(args.text_warmup_steps)
            logger.info(
                f"Text interleaving enabled: 1 text batch every {interval} steps "
                f"(ratio={args.text_mix_ratio})"
            )

    def _get_text_mix_batch():
        """Get next text batch, cycling through epochs."""
        nonlocal text_mix_iter, text_mix_epoch
        if text_mix_iter is None:
            if text_sampler:
                text_sampler.set_epoch(text_mix_epoch)
            text_mix_iter = iter(text_loader)
        try:
            return next(text_mix_iter)
        except StopIteration:
            text_mix_epoch += 1
            if text_sampler:
                text_sampler.set_epoch(text_mix_epoch)
            text_mix_iter = iter(text_loader)
            return next(text_mix_iter)

    while step < args.max_steps:
        if train_sampler:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            # Sync shutdown flag across ranks to prevent deadlock: if any rank
            # received a signal, all ranks must exit together.
            trace_enabled = args.debug_trace and step < 3
            if distributed:
                flag = torch.tensor([1.0 if shutdown else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                shutdown = flag.item() > 0.5
            if shutdown:
                if is_main:
                    logger.info("Shutdown signal received, saving checkpoint...")
                save_checkpoint(step, epoch)
                if use_wandb:
                    wandb.finish()
                if distributed:
                    dist.barrier()
                return

            if step >= args.max_steps:
                break

            # Interleave text batches during image training
            cur_interval = _get_text_mix_interval(step) if use_text_mix else 0
            is_text_step = cur_interval > 0 and step > 0 and step % cur_interval == 0
            if is_text_step:
                batch = _get_text_mix_batch()

            t0 = time.time()
            debug_trace(trace_enabled, f"train_loop: batch fetched at step={step}")
            q_inputs, d_inputs = batch
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            d_inputs = {k: v.to(device) for k, v in d_inputs.items()}
            t_transfer = time.time()

            optimizer.zero_grad()
            debug_trace(trace_enabled, "train_loop: optimizer.zero_grad done")

            # GradCache forward — handles DDP sync internally via no_sync pattern
            q_chunks = chunk_inputs(q_inputs, args.grad_cache_chunk)
            d_chunks = chunk_inputs(d_inputs, args.grad_cache_chunk)
            debug_trace(
                trace_enabled,
                f"train_loop: chunked q={len(q_chunks)} d={len(d_chunks)}",
            )

            if args.mode == "query-side-tune":
                if args.query_side_backward == "direct":
                    loss, accuracy = direct_loss_query_side(
                        query_model=model,
                        frozen_doc_model=doc_model,
                        q_inputs=q_inputs,
                        d_inputs=d_inputs,
                        logit_scale=logit_scale,
                        gather_enabled=True,
                        debug_enabled=trace_enabled,
                        hardness_alpha=args.hardness_alpha,
                    )
                else:
                    loss, accuracy = grad_cache_loss_query_side(
                        query_model=model,
                        frozen_doc_model=doc_model,
                        query_chunks=q_chunks,
                        doc_chunks=d_chunks,
                        logit_scale=logit_scale,
                        query_process_fn=forward_query,
                        doc_process_fn=forward_doc,
                        debug_enabled=trace_enabled,
                        hardness_alpha=args.hardness_alpha,
                    )
            else:
                loss, accuracy = grad_cache_loss(
                    model=model,
                    query_chunks=q_chunks,
                    doc_chunks=d_chunks,
                    logit_scale=logit_scale,
                    query_process_fn=forward_query,
                    doc_process_fn=forward_doc,
                    hardness_alpha=args.hardness_alpha,
                    loss_fn=active_loss_fn,
                )
            debug_trace(trace_enabled, "train_loop: backward path returned")
            t_fwdbwd = time.time()

            if args.max_grad_norm > 0:
                clip_grad_norm_(
                    list(model.parameters()) + list(logit_scale.parameters()),
                    args.max_grad_norm,
                )
                debug_trace(trace_enabled, "train_loop: clip_grad_norm done")

            optimizer.step()
            logit_scale.clamp_()  # clamp log_scale in-place after step (contrastors pattern)
            if ema is not None:
                ema.update(model)
            scheduler.step()
            debug_trace(trace_enabled, "train_loop: optimizer/scheduler step done")
            t_optim = time.time()
            step += 1
            dt = time.time() - t0

            if is_main:
                temp = 1.0 / logit_scale.log_scale.exp().item()
                log_scale_val = logit_scale.log_scale.item()
                cur_lr = scheduler.get_last_lr()[0]
                loss_val_item = loss.item()
                acc_val = accuracy.item()
                phase_tag = "[text-mix] " if is_text_step else ""
                dt_transfer = t_transfer - t0
                dt_fwdbwd = t_fwdbwd - t_transfer
                dt_optim = t_optim - t_fwdbwd
                logger.info(
                    f"{phase_tag}step={step}/{args.max_steps} loss={loss_val_item:.4f} "
                    f"acc={acc_val:.3f} lr={cur_lr:.2e} "
                    f"temp={temp:.4f} time={dt:.1f}s "
                    f"[data={dt_transfer:.1f}s fwd+bwd={dt_fwdbwd:.1f}s optim={dt_optim:.1f}s]"
                )
                if use_wandb:
                    grad_norm = (
                        sum(
                            p.grad.norm().item() ** 2
                            for p in model.parameters()
                            if p.requires_grad and p.grad is not None
                        )
                        ** 0.5
                    )
                    wandb.log(
                        {
                            "train/loss": loss_val_item,
                            "train/accuracy": acc_val,
                            "train/lr": cur_lr,
                            "train/temperature": temp,
                            "train/log_scale": log_scale_val,
                            "train/grad_norm": grad_norm,
                            "train/step_time_s": dt,
                            "train/epoch": epoch,
                            "train/samples_seen": step * args.batch_size * world_size,
                            "train/phase": 0.5 if is_text_step else 1,
                            "train/is_text_step": 1 if is_text_step else 0,
                        },
                        step=step,
                    )

            # Eval + Save
            # All ranks must participate in barrier; eval/test run only on rank 0
            # but we barrier before AND after to keep ranks in sync.
            need_eval = args.eval_steps > 0 and step % args.eval_steps == 0
            need_save = args.save_steps > 0 and step % args.save_steps == 0
            if args.mode == "query-side-tune":
                need_test = need_save or (
                    args.test_eval_steps > 0 and step % args.test_eval_steps == 0
                )
            else:
                need_test = (
                    args.test_eval_steps > 0 and step % args.test_eval_steps == 0
                )
            need_action = need_eval or need_test or need_save

            if need_action:
                if distributed:
                    dist.barrier()
                if need_eval and is_main:
                    torch.cuda.empty_cache()
                    run_eval(step)
                if need_save:
                    save_checkpoint(step, epoch)
                if need_test and is_main:
                    if args.mode == "query-side-tune":
                        run_query_side_tests(step)
                    elif test_datasets:
                        run_test_evals(step)
                        model.train()
                if distributed:
                    dist.barrier()

        epoch += 1
        if distributed:
            dist.barrier()

    save_checkpoint(step, epoch)
    # Final test eval — skip if last step already triggered test eval
    already_evaluated = args.test_eval_steps > 0 and step % args.test_eval_steps == 0
    if is_main and not already_evaluated:
        if args.mode == "query-side-tune":
            run_query_side_tests(step)
        elif test_datasets:
            logger.info("Final test evals...")
            run_test_evals(step)
    if is_main:
        logger.info(f"Training complete. Model saved to {args.output_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
