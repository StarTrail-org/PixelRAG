#!/usr/bin/env python3
"""Multi-GPU integration test for GradCache distributed correctness.

Tests that gather_with_grad, all_reduce(AVG), and no_sync
produce correct gradients by comparing against a full-memory reference
that uses the same distributed primitives.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
        training/tests/test_grad_multi_gpu.py

Requires 2 GPUs with ~10GB free each.
"""

import copy
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_contrastors import (
    LogitScale,
    clip_loss,
    grad_cache_loss,
    gather_with_grad,
    chunk_inputs,
    forward_query,
    forward_doc,
    _clear_rope_deltas,
)


def multi_gpu_reference(model, q_chunks, d_chunks, logit_scale):
    """Full-memory reference: forward all chunks with grad, gather, loss, backward.

    Uses the same distributed primitives (gather_with_grad, clip_loss with
    gather_enabled=True) as grad_cache_loss, but keeps all activations in
    memory for a single backward pass.
    """
    q_embs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk in q_chunks:
            _clear_rope_deltas(model)
            q_embs.append(model(**chunk))
    q_emb = torch.cat(q_embs, dim=0)

    d_embs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk in d_chunks:
            _clear_rope_deltas(model)
            d_embs.append(model(**chunk))
    d_emb = torch.cat(d_embs, dim=0)

    # Same loss as grad_cache_loss: gather docs across ranks
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, acc = clip_loss(q_emb, d_emb, logit_scale, gather_enabled=True)

    loss.backward()

    # Manual all_reduce to match grad_cache_loss behavior
    # (reference doesn't use DDP, so no automatic sync)
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for param in logit_scale.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    return loss.detach(), acc.detach()


def make_rank_data(processor, batch_size, device, rank):
    """Create per-rank fake data (different data on each rank)."""
    from PIL import Image
    import numpy as np

    # Different seed per rank → different data
    rng = np.random.RandomState(1000 + rank)
    queries = [f"Rank {rank} query about topic {i}" for i in range(batch_size)]
    images = []
    for i in range(batch_size):
        arr = rng.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))

    from train_contrastors import process_queries, process_doc_images

    query_inputs = process_queries(processor, queries)
    doc_inputs = process_doc_images(processor, images)

    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
    return query_inputs, doc_inputs


def collect_grads(model, logit_scale):
    names, grads = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            grads.append(p.grad.clone().float() if p.grad is not None else None)
    for n, p in logit_scale.named_parameters():
        names.append(f"logit_scale.{n}")
        grads.append(p.grad.clone().float() if p.grad is not None else None)
    return names, grads


def compare_grads(grads_a, grads_b, names):
    """Compare gradients, return cosine similarity."""
    flat_a, flat_b = [], []
    for name, ga, gb in zip(names, grads_a, grads_b):
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            return 0.0  # mismatch
        if ga.abs().max().item() == 0 and gb.abs().max().item() == 0:
            continue
        flat_a.append(ga.flatten())
        flat_b.append(gb.flatten())

    if not flat_a:
        return 1.0

    a = torch.cat(flat_a)
    b = torch.cat(flat_b)
    cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    rel_l2 = (a - b).norm().item() / max(a.norm().item(), 1e-12)
    return cosine, rel_l2


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    assert world_size == 2, f"This test requires exactly 2 GPUs, got {world_size}"

    if is_main:
        print(f"Multi-GPU gradient test: {world_size} GPUs")

    # Load model
    from models.biqwen3 import BiQwen3
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model

    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    base_model = BiQwen3.from_pretrained(model_name, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_name)
    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    tile = patch_size * merge_size
    processor.image_processor.max_pixels = 256 * tile * tile
    processor.image_processor.size["longest_edge"] = (
        processor.image_processor.max_pixels
    )
    processor.tokenizer.padding_side = "left"

    batch_size = 4
    chunk_size = 2
    results = {}

    # =========================================================================
    # Test 1: GradCache multi-GPU vs full-memory reference (no dropout)
    # =========================================================================
    if is_main:
        print("\n" + "=" * 80)
        print("TEST 1: GradCache multi-GPU vs reference (no dropout)")
        print("=" * 80)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(copy.deepcopy(base_model), lora_config).to(device)
    model.train()

    # Broadcast model state from rank 0 to ensure all ranks start identical
    model_state = model.state_dict()
    for key in model_state:
        dist.broadcast(model_state[key], src=0)
    model.load_state_dict(model_state)
    model_state = copy.deepcopy(model_state)
    ls_state = LogitScale(init_value=1 / 0.07).state_dict()

    # Per-rank data (each rank gets different queries + images)
    query_inputs, doc_inputs = make_rank_data(processor, batch_size, device, rank)

    # --- Reference ---
    model.load_state_dict(model_state)
    ls_ref = LogitScale(init_value=1 / 0.07).to(device)
    ls_ref.load_state_dict(ls_state)
    model.zero_grad()
    ls_ref.zero_grad()

    q_chunks_ref = chunk_inputs(
        {k: v.clone() for k, v in query_inputs.items()}, chunk_size
    )
    d_chunks_ref = chunk_inputs(
        {k: v.clone() for k, v in doc_inputs.items()}, chunk_size
    )

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    loss_ref, _ = multi_gpu_reference(model, q_chunks_ref, d_chunks_ref, ls_ref)
    names, grads_ref = collect_grads(model, ls_ref)

    # --- GradCache with DDP ---
    model.load_state_dict(model_state)
    ls_gc = LogitScale(init_value=1 / 0.07).to(device)
    ls_gc.load_state_dict(ls_state)
    model.zero_grad()
    ls_gc.zero_grad()

    # Wrap in DDP (same as train_contrastors.py)
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    q_chunks_gc = chunk_inputs(
        {k: v.clone() for k, v in query_inputs.items()}, chunk_size
    )
    d_chunks_gc = chunk_inputs(
        {k: v.clone() for k, v in doc_inputs.items()}, chunk_size
    )

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    loss_gc, acc_gc = grad_cache_loss(
        model=model_ddp,
        query_chunks=q_chunks_gc,
        doc_chunks=d_chunks_gc,
        logit_scale=ls_gc,
        query_process_fn=forward_query,
        doc_process_fn=forward_doc,
    )
    _, grads_gc = collect_grads(model, ls_gc)  # grads are on the raw model

    cosine, rel_l2 = compare_grads(grads_ref, grads_gc, names)
    if is_main:
        print(f"  Loss ref:  {loss_ref.item():.8f}")
        print(f"  Loss gc:   {loss_gc.item():.8f}")
        print(f"  Loss diff: {abs(loss_ref.item() - loss_gc.item()):.2e}")
        print(f"  Cosine:    {cosine:.10f}")
        print(f"  Rel L2:    {rel_l2:.6e}")
        print(f"  [{'PASS' if cosine > 0.999 else 'FAIL'}]")
    results["T1_multi_gpu_no_dropout"] = cosine

    # Cleanup DDP for next test
    del model_ddp

    # =========================================================================
    # Test 2: GradCache multi-GPU with dropout (tests RandContext + DDP)
    # =========================================================================
    if is_main:
        print("\n" + "=" * 80)
        print("TEST 2: GradCache multi-GPU vs reference (with dropout)")
        print("=" * 80)

    lora_drop = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
    )
    model2 = get_peft_model(copy.deepcopy(base_model), lora_drop).to(device)
    model2.train()

    m2_state = model2.state_dict()
    for key in m2_state:
        dist.broadcast(m2_state[key], src=0)
    model2.load_state_dict(m2_state)
    m2_state = copy.deepcopy(m2_state)

    # --- Reference ---
    model2.load_state_dict(m2_state)
    ls_ref2 = LogitScale(init_value=1 / 0.07).to(device)
    ls_ref2.load_state_dict(ls_state)
    model2.zero_grad()
    ls_ref2.zero_grad()

    q_ref2 = chunk_inputs({k: v.clone() for k, v in query_inputs.items()}, chunk_size)
    d_ref2 = chunk_inputs({k: v.clone() for k, v in doc_inputs.items()}, chunk_size)

    torch.manual_seed(99 + rank)
    torch.cuda.manual_seed_all(99 + rank)
    loss_ref2, _ = multi_gpu_reference(model2, q_ref2, d_ref2, ls_ref2)
    names2, grads_ref2 = collect_grads(model2, ls_ref2)

    # --- GradCache with DDP ---
    model2.load_state_dict(m2_state)
    ls_gc2 = LogitScale(init_value=1 / 0.07).to(device)
    ls_gc2.load_state_dict(ls_state)
    model2.zero_grad()
    ls_gc2.zero_grad()

    model2_ddp = torch.nn.parallel.DistributedDataParallel(
        model2,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    q_gc2 = chunk_inputs({k: v.clone() for k, v in query_inputs.items()}, chunk_size)
    d_gc2 = chunk_inputs({k: v.clone() for k, v in doc_inputs.items()}, chunk_size)

    torch.manual_seed(99 + rank)
    torch.cuda.manual_seed_all(99 + rank)
    loss_gc2, _ = grad_cache_loss(
        model=model2_ddp,
        query_chunks=q_gc2,
        doc_chunks=d_gc2,
        logit_scale=ls_gc2,
        query_process_fn=forward_query,
        doc_process_fn=forward_doc,
    )
    _, grads_gc2 = collect_grads(model2, ls_gc2)

    cosine2, rel_l2_2 = compare_grads(grads_ref2, grads_gc2, names2)
    if is_main:
        print(f"  Loss ref:  {loss_ref2.item():.8f}")
        print(f"  Loss gc:   {loss_gc2.item():.8f}")
        print(f"  Loss diff: {abs(loss_ref2.item() - loss_gc2.item()):.2e}")
        print(f"  Cosine:    {cosine2:.10f}")
        print(f"  Rel L2:    {rel_l2_2:.6e}")
        print(f"  [{'PASS' if cosine2 > 0.999 else 'FAIL'}]")
    results["T2_multi_gpu_dropout"] = cosine2

    del model2_ddp

    # =========================================================================
    # Test 3: Verify gather_with_grad backward is correct
    # =========================================================================
    if is_main:
        print("\n" + "=" * 80)
        print("TEST 3: gather_with_grad backward correctness")
        print("=" * 80)

    # Each rank has a vector x_r. gather_with_grad → gathered = [x_0, x_1, ...].
    # Each rank independently computes loss = gathered.sum().
    # Backward of all_gather is reduce_scatter: sums grad from all ranks → each rank.
    # Since each rank sends grad=1 for ALL gathered elements, each rank receives
    # sum of W contributions = W for its own piece.
    x = torch.randn(3, 4, device=device, requires_grad=True)

    gathered = gather_with_grad(x)
    assert gathered.shape == (3 * world_size, 4), (
        f"Expected ({3 * world_size},4), got {gathered.shape}"
    )

    loss = gathered.sum()
    loss.backward()

    # Expected: world_size (each rank contributes gradient=1 for every element,
    # reduce_scatter sums them, so each rank gets W*1 for its own slice)
    expected_grad = torch.full_like(x, world_size)
    grad_ok = torch.allclose(x.grad, expected_grad, atol=1e-5)
    if is_main:
        print(f"  gathered.shape: {gathered.shape}")
        print(f"  x.grad mean: {x.grad.mean().item():.1f}  (expected: {world_size}.0)")
        print(f"  [{'PASS' if grad_ok else 'FAIL'}]")
    results["T3_gather_backward"] = 1.0 if grad_ok else 0.0

    # =========================================================================
    # Test 4: loss * world_size + all_reduce(AVG) = correct total gradient
    # =========================================================================
    if is_main:
        print("\n" + "=" * 80)
        print("TEST 4: loss * W + all_reduce(AVG) gives gradient of total loss")
        print("=" * 80)

    # Simulates the contrastors loss scaling convention:
    # - All ranks share parameter θ (same model weights)
    # - Each rank r computes local_loss_r(θ) using its own data
    # - Gradient: W * d(local_loss_r)/d(θ), then all_reduce(AVG)
    # - Result: (1/W) * Σ_r [W * d(L_r)/d(θ)] = Σ_r d(L_r)/d(θ) = d(total_L)/d(θ)
    #
    # We verify: all_reduce(AVG) of (W * local_grad) = sum of all local grads

    # Shared parameter θ (same on all ranks)
    torch.manual_seed(500)
    theta = torch.randn(4, 4, device=device, requires_grad=True)

    # Per-rank data
    torch.manual_seed(600 + rank)
    data_r = torch.randn(4, 4, device=device)

    local_loss = (theta * data_r).sum()
    scaled = local_loss * world_size
    scaled.backward()
    grad_scaled = theta.grad.clone()  # = W * data_r

    # all_reduce(AVG): gives (1/W) * Σ(W * data_r) = Σ data_r
    dist.all_reduce(grad_scaled, op=dist.ReduceOp.AVG)

    # Expected: Σ data_r = sum of all ranks' data
    # Compute by all_reducing the data itself
    all_data_sum = data_r.clone()
    dist.all_reduce(all_data_sum, op=dist.ReduceOp.SUM)

    t4_ok = torch.allclose(grad_scaled, all_data_sum, atol=1e-5)
    if is_main:
        print(
            f"  grad after loss*W + all_reduce(AVG): mean={grad_scaled.mean().item():.6f}"
        )
        print(
            f"  expected (Σ data_r):                 mean={all_data_sum.mean().item():.6f}"
        )
        print(f"  match: {t4_ok}")
        print(f"  [{'PASS' if t4_ok else 'FAIL'}]")
    results["T4_loss_scaling"] = 1.0 if t4_ok else 0.0

    # =========================================================================
    # Test 5: Gradients are identical across ranks after all_reduce
    # =========================================================================
    if is_main:
        print("\n" + "=" * 80)
        print("TEST 5: Gradients identical across ranks after GradCache")
        print("=" * 80)

    # Re-use grads from Test 1: check that rank 0 and rank 1 have same grads
    # (all_reduce should have synced them)
    if grads_gc:
        # Gather a gradient tensor from each rank to rank 0
        sample_grad = grads_gc[0]  # first param's gradient
        if sample_grad is not None:
            # All-gather the gradient from all ranks
            gathered_grads = [torch.zeros_like(sample_grad) for _ in range(world_size)]
            dist.all_gather(gathered_grads, sample_grad)
            if is_main:
                # Compare rank 0 vs rank 1 gradients
                diff = (gathered_grads[0] - gathered_grads[1]).abs().max().item()
                t5_ok = diff < 1e-6
                print(f"  Max gradient diff between rank 0 and rank 1: {diff:.2e}")
                print(f"  [{'PASS' if t5_ok else 'FAIL'}]")
                results["T5_grad_sync"] = 1.0 if t5_ok else 0.0
        else:
            if is_main:
                print("  No gradients to compare")
                results["T5_grad_sync"] = 0.0
    else:
        if is_main:
            results["T5_grad_sync"] = 0.0

    # =========================================================================
    # Summary
    # =========================================================================
    if is_main:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        cos_threshold = 0.999
        all_pass = True
        for name, val in results.items():
            if name.startswith("T1") or name.startswith("T2"):
                ok = val > cos_threshold
                print(f"  {name:<35} cosine={val:.10f}  [{'PASS' if ok else 'FAIL'}]")
            else:
                ok = val > 0.5
                print(f"  {name:<35} [{'PASS' if ok else 'FAIL'}]")
            if not ok:
                all_pass = False

        if all_pass:
            print("\n  ALL TESTS PASSED.")
            print("  Multi-GPU GradCache is correct:")
            print("  - gather_with_grad backward ✓")
            print("  - loss*W + all_reduce(AVG) = correct gradient ✓")
            print("  - GradCache matches reference on 2 GPUs ✓")
            print("  - RandContext works with DDP ✓")
            print("  - Gradients synced across ranks ✓")
        else:
            print("\n  SOME TESTS FAILED.")
        print("=" * 80)

    # Compute pass/fail on all ranks (results dict is only populated on main)
    if is_main:
        cos_threshold = 0.999
        all_pass = True
        for name, val in results.items():
            if name.startswith("T1") or name.startswith("T2"):
                if val <= cos_threshold:
                    all_pass = False
            else:
                if val <= 0.5:
                    all_pass = False
        pass_flag = torch.tensor([1.0 if all_pass else 0.0], device=device)
    else:
        pass_flag = torch.tensor([0.0], device=device)

    dist.broadcast(pass_flag, src=0)
    final_pass = pass_flag.item() > 0.5

    dist.destroy_process_group()
    return 0 if final_pass else 1


if __name__ == "__main__":
    sys.exit(main())
