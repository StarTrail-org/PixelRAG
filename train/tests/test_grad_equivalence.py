#!/usr/bin/env python3
"""Verify GradCache produces identical gradients to full-memory chunked reference.

The CORRECT comparison for GradCache is NOT "vanilla full-batch" vs "GradCache chunked",
because chunked forward can produce slightly different embeddings (bf16 accumulation order).

Instead, we compare:
  1. REFERENCE: forward each chunk WITH grad (same chunks!) → concat embeddings →
     compute loss → backward. This keeps all activations in memory.
  2. GRADCACHE: the actual 3-step process from train_contrastors.py.

Both process the SAME chunks in the SAME order with the SAME precision, so the
embeddings are identical. The only difference is HOW gradients are computed:
reference does one backward through the full graph, GradCache uses surrogate loss.
If GradCache's chain rule decomposition is correct, gradients must match exactly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python training/tests/test_grad_equivalence.py
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_contrastors import (
    LogitScale,
    _clear_rope_deltas,
    chunk_inputs,
    clip_loss,
    forward_doc,
    forward_query,
    grad_cache_loss,
)


def chunked_reference_forward_backward(model, q_chunks, d_chunks, logit_scale):
    """Full-memory reference: forward ALL chunks with grad, then single backward.

    This computes the EXACT SAME embeddings as GradCache step 1 (same chunks,
    same order, same autocast), but keeps all activations for a single backward.
    """
    # Forward each chunk with grad, same as GradCache would
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

    # Same loss computation as GradCache step 2
    # gather_enabled=True with no dist → effectively gather_enabled=False
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, acc = clip_loss(q_emb, d_emb, logit_scale, gather_enabled=True)

    loss.backward()
    return loss.detach(), acc.detach()


def make_fake_inputs(processor, batch_size, device):
    import numpy as np
    from PIL import Image

    queries = [f"What is topic number {i}?" for i in range(batch_size)]
    images = []
    for i in range(batch_size):
        arr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))

    from train_contrastors import process_doc_images, process_queries

    query_inputs = process_queries(processor, queries)
    doc_inputs = process_doc_images(processor, images)

    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
    return query_inputs, doc_inputs


def collect_grads(model, logit_scale):
    """Collect gradients, cast to fp32 for comparison."""
    names, grads = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            grads.append(p.grad.clone().float() if p.grad is not None else None)
    for n, p in logit_scale.named_parameters():
        names.append(f"logit_scale.{n}")
        grads.append(p.grad.clone().float() if p.grad is not None else None)
    return names, grads


def compare_gradients(grads_ref, grads_gc, names, verbose=True):
    """Compare gradients. Returns (cosine, rel_l2, max_rel_diff)."""
    flat_ref, flat_gc = [], []
    max_rel = 0.0
    n_compared = 0

    if verbose:
        print(f"\n{'Parameter':<60} {'MaxDiff':>12} {'MeanDiff':>12} {'RelDiff':>12}")
        print("-" * 100)

    for name, gr, ggc in zip(names, grads_ref, grads_gc):
        if gr is None and ggc is None:
            continue
        if gr is None or ggc is None:
            print(f"  MISMATCH: {name} — one grad is None")
            return 0.0, float("inf"), float("inf")
        if gr.abs().max().item() == 0 and ggc.abs().max().item() == 0:
            continue

        flat_ref.append(gr.flatten())
        flat_gc.append(ggc.flatten())
        n_compared += 1

        abs_diff = (gr - ggc).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        scale = gr.abs().mean().item()
        rel_diff = mean_diff / max(scale, 1e-12)
        max_rel = max(max_rel, rel_diff)

        if verbose:
            print(f"{name:<60} {max_diff:>12.6e} {mean_diff:>12.6e} {rel_diff:>12.6e}")

    if not flat_ref:
        print("No non-zero gradients to compare.")
        return 1.0, 0.0, 0.0

    ref_cat = torch.cat(flat_ref)
    gc_cat = torch.cat(flat_gc)
    cosine = F.cosine_similarity(ref_cat.unsqueeze(0), gc_cat.unsqueeze(0)).item()
    l2_diff = (ref_cat - gc_cat).norm().item()
    rel_l2 = l2_diff / max(ref_cat.norm().item(), 1e-12)

    if verbose:
        print(f"\nCompared {n_compared} parameter groups")
        print(f"  Cosine similarity:  {cosine:.10f}")
        print(f"  Relative L2 diff:   {rel_l2:.6e}")
        print(f"  Max per-param rel:  {max_rel:.6e}")

    return cosine, rel_l2, max_rel


def run_test(
    label,
    model,
    model_state,
    ls_state,
    query_inputs,
    doc_inputs,
    chunk_size,
    device,
    verbose=True,
):
    """Compare chunked-reference vs GradCache for a given chunk_size."""

    # --- Chunked reference (full-memory backward) ---
    model.load_state_dict(model_state)
    ls_ref = LogitScale(init_value=1 / 0.07).to(device)
    ls_ref.load_state_dict(ls_state)
    model.zero_grad()
    ls_ref.zero_grad()

    # Create fresh chunks for each path (can't share autograd graphs)
    q_chunks_ref = chunk_inputs(
        {k: v.clone() for k, v in query_inputs.items()}, chunk_size
    )
    d_chunks_ref = chunk_inputs(
        {k: v.clone() for k, v in doc_inputs.items()}, chunk_size
    )

    # Seed before reference forward so dropout masks are deterministic
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loss_ref, _ = chunked_reference_forward_backward(
        model, q_chunks_ref, d_chunks_ref, ls_ref
    )
    names, grads_ref = collect_grads(model, ls_ref)

    # --- GradCache (actual function from train_contrastors.py) ---
    model.load_state_dict(model_state)
    ls_gc = LogitScale(init_value=1 / 0.07).to(device)
    ls_gc.load_state_dict(ls_state)
    model.zero_grad()
    ls_gc.zero_grad()

    q_chunks_gc = chunk_inputs(
        {k: v.clone() for k, v in query_inputs.items()}, chunk_size
    )
    d_chunks_gc = chunk_inputs(
        {k: v.clone() for k, v in doc_inputs.items()}, chunk_size
    )

    # Same seed → GradCache step 1 (no-grad forward) uses the same dropout masks
    # as the reference. Step 3 (replay) uses RandContext to reproduce them.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loss_gc, acc_gc = grad_cache_loss(
        model=model,
        query_chunks=q_chunks_gc,
        doc_chunks=d_chunks_gc,
        logit_scale=ls_gc,
        query_process_fn=forward_query,
        doc_process_fn=forward_doc,
    )
    _, grads_gc = collect_grads(model, ls_gc)

    if verbose:
        print(f"\n  Loss reference: {loss_ref.item():.8f}")
        print(f"  Loss GradCache: {loss_gc.item():.8f}")
        print(f"  Loss diff:      {abs(loss_ref.item() - loss_gc.item()):.2e}")

    cosine, rel_l2, max_rel = compare_gradients(
        grads_ref, grads_gc, names, verbose=verbose
    )
    return cosine, rel_l2, loss_ref.item(), loss_gc.item()


def main():
    device = torch.device("cuda:0")
    batch_size = 4

    print("Loading model + processor...")
    from models.biqwen3 import BiQwen3
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

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

    print("Creating fake data...")
    query_inputs, doc_inputs = make_fake_inputs(processor, batch_size, device)

    results = {}

    # =========================================================================
    # PART A: dropout=0 (isolates GradCache chain-rule math)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART A: lora_dropout=0.0 — tests GradCache chain-rule decomposition")
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
    model_state = copy.deepcopy(model.state_dict())
    ls_state = LogitScale(init_value=1 / 0.07).state_dict()

    for cs in [batch_size, 2, 1]:
        label = f"A_chunk{cs}"
        print(f"\n--- chunk_size={cs} {'(degenerate)' if cs == batch_size else ''} ---")
        cos, rl2, _, _ = run_test(
            label, model, model_state, ls_state, query_inputs, doc_inputs, cs, device
        )
        results[label] = cos

    # =========================================================================
    # PART B: dropout=0.05 — tests RandContext RNG replay
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: lora_dropout=0.05 — tests RandContext RNG state replay")
    print("=" * 80)

    lora_config_drop = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
    )
    model_drop = get_peft_model(copy.deepcopy(base_model), lora_config_drop).to(device)
    model_drop.train()
    model_drop_state = copy.deepcopy(model_drop.state_dict())

    for cs in [batch_size, 2, 1]:
        label = f"B_drop_chunk{cs}"
        print(
            f"\n--- chunk_size={cs} with dropout {'(degenerate)' if cs == batch_size else ''} ---"
        )
        cos, rl2, _, _ = run_test(
            label,
            model_drop,
            model_drop_state,
            ls_state,
            query_inputs,
            doc_inputs,
            cs,
            device,
        )
        results[label] = cos

    # =========================================================================
    # PART C: clip_loss label correctness (unit tests, no model needed)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART C: clip_loss label arithmetic")
    print("=" * 80)

    # C1: Basic case — no hard negatives, no gather
    print("\n--- Test C1: basic labels (no hard negs, no gather) ---")
    N, D = 4, 8
    q = F.normalize(torch.randn(N, D, device=device), dim=-1)
    d = F.normalize(torch.randn(N, D, device=device), dim=-1)
    ls = LogitScale(init_value=1 / 0.07).to(device)
    loss, acc = clip_loss(q, d, ls, gather_enabled=False)
    # Labels should be [0, 1, 2, 3] — each query matches its own doc
    sim = ls(q @ d.T)
    expected_labels = torch.arange(N, device=device)
    expected_loss = F.cross_entropy(sim, expected_labels)
    c1_ok = abs(loss.item() - expected_loss.item()) < 1e-5
    print(
        f"  loss={loss.item():.6f}  expected={expected_loss.item():.6f}  "
        f"diff={abs(loss.item() - expected_loss.item()):.2e}  [{'PASS' if c1_ok else 'FAIL'}]"
    )
    results["C1_basic_labels"] = 1.0 if c1_ok else 0.0

    # C2: Hard negatives — 2 hard negs per query, docs interleaved
    print("\n--- Test C2: hard negative labels (2 negs/query) ---")
    num_hard_neg = 2
    docs_per_q = 1 + num_hard_neg  # 3
    # doc layout: [pos0, neg0a, neg0b, pos1, neg1a, neg1b, pos2, ..., pos3, ...]
    d_hn = F.normalize(torch.randn(N * docs_per_q, D, device=device), dim=-1)
    loss_hn, _ = clip_loss(q, d_hn, ls, gather_enabled=False)
    # Labels should point to positive positions: [0, 3, 6, 9]
    sim_hn = ls(q @ d_hn.T)
    expected_hn_labels = torch.arange(N, device=device) * docs_per_q
    expected_hn_loss = F.cross_entropy(sim_hn, expected_hn_labels)
    c2_ok = abs(loss_hn.item() - expected_hn_loss.item()) < 1e-5
    print(
        f"  labels={expected_hn_labels.tolist()}  loss={loss_hn.item():.6f}  "
        f"expected={expected_hn_loss.item():.6f}  "
        f"diff={abs(loss_hn.item() - expected_hn_loss.item()):.2e}  [{'PASS' if c2_ok else 'FAIL'}]"
    )
    results["C2_hard_neg_labels"] = 1.0 if c2_ok else 0.0

    # C3: Verify assertion fires for bad divisibility
    print("\n--- Test C3: assertion for bad doc/query ratio ---")
    d_bad = F.normalize(torch.randn(N + 1, D, device=device), dim=-1)
    c3_ok = False
    try:
        clip_loss(q, d_bad, ls, gather_enabled=False)
        print("  ERROR: no assertion raised!")
    except AssertionError as e:
        c3_ok = True
        print(f"  AssertionError raised as expected: {e}")
        print("  [PASS]")
    results["C3_divisibility_assert"] = 1.0 if c3_ok else 0.0

    # =========================================================================
    # PART D: _clear_rope_deltas prevents stale state
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART D: _clear_rope_deltas prevents image→text state leakage")
    print("=" * 80)

    # Use the no-dropout model for this test
    model.load_state_dict(model_state)
    model.eval()

    # Forward an image batch to populate rope_deltas
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _clear_rope_deltas(model)
            _ = model(**{k: v[:2] for k, v in doc_inputs.items()})

    # Check that rope_deltas is set on the inner model
    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    has_rope = hasattr(inner, "rope_deltas") and inner.rope_deltas is not None
    print(f"  rope_deltas set after image forward: {has_rope}")

    # Now forward text — WITHOUT clearing, this should fail or give wrong results
    # if rope_deltas shape mismatches the text batch
    # First, demonstrate that clearing prevents the issue:
    _clear_rope_deltas(model)
    cleared = not hasattr(inner, "rope_deltas") or inner.rope_deltas is None
    print(f"  rope_deltas cleared after _clear_rope_deltas: {cleared}")

    d_ok = False
    try:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _clear_rope_deltas(model)
                _ = model(**{k: v[:3] for k, v in query_inputs.items()})
        d_ok = True
        print("  Text forward after clear succeeded: [PASS]")
    except Exception as e:
        print(f"  Text forward after clear failed: {e}  [FAIL]")
    results["D_rope_deltas"] = 1.0 if d_ok else 0.0

    model.train()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Threshold: the reference and GradCache should produce nearly identical gradients
    # since they compute the same function. Only bf16 non-determinism can cause diffs.
    cos_threshold = 0.999

    all_pass = True
    for name, cos in results.items():
        ok = cos > cos_threshold
        if not ok:
            all_pass = False
        print(f"  {name:<25} cosine={cos:.10f}  [{'PASS' if ok else 'FAIL'}]")

    print(f"\n  Threshold: cosine > {cos_threshold}")
    if all_pass:
        print("\n  ALL TESTS PASSED.")
        print("  GradCache's surrogate backward matches full-memory backward.")
        print("  RandContext correctly replays dropout RNG states.")
    else:
        print("\n  SOME TESTS FAILED — GradCache produces incorrect gradients.")
        for name, cos in results.items():
            if cos <= cos_threshold:
                print(f"    {name}: cosine={cos:.10f}")
    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
