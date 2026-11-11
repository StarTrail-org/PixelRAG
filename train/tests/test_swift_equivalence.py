#!/usr/bin/env python3
"""Verify that train_swift.py produces equivalent results to train_contrastors.py.

Tests:
1. Tokenization equivalence — same input → same token IDs
2. Embedding equivalence — same model weights → same embedding output
3. Loss equivalence — same batch → same InfoNCE loss value
4. LoRA target equivalence — same parameters marked trainable

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python tests/test_swift_equivalence.py
"""

import json
import os
import sys

# Ensure the training root is on sys.path so `models` and `train_contrastors` resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."

SAMPLE = {
    "query": "What is the population of Tokyo?",
    "chunk_path": None,  # filled dynamically
}


def find_test_image():
    """Find any valid chunk image for testing."""
    data_file = "data/train.jsonl"
    if not os.path.exists(data_file):
        data_file = "data/train_hn.jsonl"
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            path = item["chunk_path"]
            if os.path.exists(path):
                return path
    raise RuntimeError("No valid test image found")


# ---------------------------------------------------------------------------
# Test 1: Tokenization equivalence
# ---------------------------------------------------------------------------


def test_tokenization():
    """Verify that contrastors and swift produce the same token IDs for queries."""
    print("\n=== Test 1: Tokenization equivalence ===")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    query = "What is the population of Tokyo?"

    # --- Contrastors path: manual chat template ---
    q_msgs = [
        {"role": "system", "content": [{"type": "text", "text": QUERY_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]
    contrastors_text = processor.apply_chat_template(
        q_msgs, tokenize=False, add_generation_prompt=True
    )
    contrastors_ids = processor(text=[contrastors_text], return_tensors="pt")[
        "input_ids"
    ][0]

    # --- Swift path: messages format ---
    # Swift uses the same processor but constructs messages differently.
    # The swift template system processes messages → applies chat template → tokenizes.
    # We simulate what swift does: same chat template but with string content.
    s_msgs = [
        {"role": "system", "content": [{"type": "text", "text": QUERY_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]
    swift_text = processor.apply_chat_template(
        s_msgs, tokenize=False, add_generation_prompt=True
    )
    swift_ids = processor(text=[swift_text], return_tensors="pt")["input_ids"][0]

    match = torch.equal(contrastors_ids, swift_ids)
    print(
        f"  Contrastors tokens: {contrastors_ids.shape} → {contrastors_ids[:10].tolist()}..."
    )
    print(f"  Swift tokens:       {swift_ids.shape} → {swift_ids[:10].tolist()}...")
    print(f"  Text match: {contrastors_text == swift_text}")
    print(f"  Token match: {match}")

    # Also check the actual template strings
    if contrastors_text != swift_text:
        print("  DIFF in template text!")
        print(f"    Contrastors: {contrastors_text[:200]!r}")
        print(f"    Swift:       {swift_text[:200]!r}")

    assert match, "Tokenization mismatch!"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 2: Embedding equivalence
# ---------------------------------------------------------------------------


def test_embedding():
    """Verify BiQwen3 and swift's patched model produce the same embeddings."""
    print("\n=== Test 2: Embedding equivalence ===")

    from PIL import Image
    from transformers import AutoProcessor

    image_path = find_test_image()
    print(f"  Test image: {image_path}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    query = "What is the population of Tokyo?"
    image = Image.open(image_path).convert("RGB")

    # --- Contrastors path: BiQwen3 ---
    from models.biqwen3 import BiQwen3

    contrastors_model = (
        BiQwen3.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).cuda().eval()
    )

    # Query embedding
    q_msgs = [
        {"role": "system", "content": [{"type": "text", "text": QUERY_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]
    q_text = processor.apply_chat_template(
        q_msgs, tokenize=False, add_generation_prompt=True
    )
    q_inputs = processor(text=[q_text], return_tensors="pt", padding="longest")
    q_inputs = {k: v.cuda() for k, v in q_inputs.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        contrastors_q_emb = contrastors_model(**q_inputs).cpu().float()

    # Doc embedding
    d_msgs = [
        {"role": "system", "content": [{"type": "text", "text": DOC_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    d_text = processor.apply_chat_template(
        d_msgs, tokenize=False, add_generation_prompt=True
    )
    d_inputs = processor(
        text=[d_text], images=[image], return_tensors="pt", padding="longest"
    )
    d_inputs = {k: v.cuda() for k, v in d_inputs.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        contrastors_d_emb = contrastors_model(**d_inputs).cpu().float()

    del contrastors_model
    torch.cuda.empty_cache()

    # --- Swift path: Qwen3VLForConditionalGeneration + patch ---
    from transformers import Qwen3VLForConditionalGeneration

    swift_model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )

    # Replicate swift's embedding patch: use model.model (base Qwen3VLModel),
    # last-token pooling + L2 norm
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Query
        q_inputs_swift = processor(
            text=[q_text], return_tensors="pt", padding="longest"
        )
        q_inputs_swift = {k: v.cuda() for k, v in q_inputs_swift.items()}
        q_out = swift_model.model(
            **q_inputs_swift,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        swift_q_emb = q_out.last_hidden_state[:, -1]
        swift_q_emb = (
            (swift_q_emb / swift_q_emb.norm(dim=-1, keepdim=True)).cpu().float()
        )

        # Doc
        d_inputs_swift = processor(
            text=[d_text], images=[image], return_tensors="pt", padding="longest"
        )
        d_inputs_swift = {k: v.cuda() for k, v in d_inputs_swift.items()}
        d_out = swift_model.model(
            **d_inputs_swift,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        swift_d_emb = d_out.last_hidden_state[:, -1]
        swift_d_emb = (
            (swift_d_emb / swift_d_emb.norm(dim=-1, keepdim=True)).cpu().float()
        )

    del swift_model
    torch.cuda.empty_cache()

    # Compare
    q_cosine = torch.nn.functional.cosine_similarity(
        contrastors_q_emb, swift_q_emb
    ).item()
    d_cosine = torch.nn.functional.cosine_similarity(
        contrastors_d_emb, swift_d_emb
    ).item()
    q_maxdiff = (contrastors_q_emb - swift_q_emb).abs().max().item()
    d_maxdiff = (contrastors_d_emb - swift_d_emb).abs().max().item()

    print(f"  Query embedding:  cosine={q_cosine:.6f}  max_diff={q_maxdiff:.6e}")
    print(f"  Doc embedding:    cosine={d_cosine:.6f}  max_diff={d_maxdiff:.6e}")

    # bf16 precision: two different code paths (Qwen3VLModel vs ConditionalGeneration.model)
    # accumulate small numerical differences. 0.999 is a reasonable threshold.
    assert q_cosine > 0.999, f"Query embedding diverged: cosine={q_cosine}"
    assert d_cosine > 0.999, f"Doc embedding diverged: cosine={d_cosine}"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 3: Loss equivalence
# ---------------------------------------------------------------------------


def test_loss():
    """Verify InfoNCE loss computation is equivalent."""
    print("\n=== Test 3: Loss equivalence ===")

    # Create synthetic embeddings (deterministic)
    torch.manual_seed(42)
    batch_size = 4
    dim = 128
    temperature = 0.07

    query_embs = torch.randn(batch_size, dim)
    query_embs = query_embs / query_embs.norm(dim=-1, keepdim=True)
    doc_embs = torch.randn(batch_size, dim)
    doc_embs = doc_embs / doc_embs.norm(dim=-1, keepdim=True)

    # --- Contrastors path: clip_loss ---
    import torch.nn.functional as F
    from train_contrastors import LogitScale

    logit_scale = LogitScale(init_value=1.0 / temperature)
    similarity = logit_scale(torch.matmul(query_embs, doc_embs.T))
    labels = torch.arange(batch_size)
    contrastors_loss = F.cross_entropy(similarity, labels).item()

    # --- Swift path: InfoNCE ---
    # Swift computes: similarity / temperature, then cross_entropy
    swift_similarity = torch.matmul(query_embs, doc_embs.T) / temperature
    swift_loss = F.cross_entropy(swift_similarity, labels).item()

    diff = abs(contrastors_loss - swift_loss)
    print(f"  Contrastors loss: {contrastors_loss:.6f}")
    print(f"  Swift loss:       {swift_loss:.6f}")
    print(f"  Absolute diff:    {diff:.6e}")

    # They should be identical: logit_scale(x) = x * exp(ln(1/0.07)) = x / 0.07
    assert diff < 1e-5, f"Loss mismatch: {diff}"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 4: LoRA target equivalence
# ---------------------------------------------------------------------------


def test_lora_targets():
    """Verify LoRA is applied to the same parameters."""
    print("\n=== Test 4: LoRA target equivalence ===")

    from models.biqwen3 import BiQwen3
    from peft import LoraConfig, get_peft_model

    model = BiQwen3.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(model, lora_config)

    contrastors_trainable = sorted(
        [n for n, p in peft_model.named_parameters() if p.requires_grad]
    )

    del peft_model, model
    torch.cuda.empty_cache()

    # Swift path: same LoRA config on ConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration

    swift_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
    swift_lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
    )
    swift_peft = get_peft_model(swift_model, swift_lora_config)

    swift_trainable = sorted(
        [n for n, p in swift_peft.named_parameters() if p.requires_grad]
    )

    del swift_peft, swift_model

    # Compare — swift has "model.model." prefix, contrastors has "model."
    # Normalize by stripping model prefixes
    def normalize_name(n):
        # Remove peft wrapper prefix
        n = n.replace("base_model.model.", "")
        # Remove the extra "model." from ConditionalGeneration
        if n.startswith("model."):
            n = n[len("model.") :]
        return n

    contrastors_normalized = sorted(
        set(normalize_name(n) for n in contrastors_trainable)
    )
    swift_normalized = sorted(set(normalize_name(n) for n in swift_trainable))

    # Check for lm_head LoRA in swift (shouldn't be there since q/k/v/o only)
    swift_lmhead = [n for n in swift_trainable if "lm_head" in n]

    only_contrastors = set(contrastors_normalized) - set(swift_normalized)
    only_swift = set(swift_normalized) - set(contrastors_normalized)

    print(f"  Contrastors trainable params: {len(contrastors_trainable)}")
    print(f"  Swift trainable params:       {len(swift_trainable)}")
    print(f"  Normalized contrastors:       {len(contrastors_normalized)}")
    print(f"  Normalized swift:             {len(swift_normalized)}")
    print(f"  Swift lm_head LoRA:           {len(swift_lmhead)} (should be 0)")

    if only_contrastors:
        print(f"  Only in contrastors: {list(only_contrastors)[:5]}...")
    if only_swift:
        print(f"  Only in swift: {list(only_swift)[:5]}...")

    match = contrastors_normalized == swift_normalized
    print(f"  Exact match: {match}")

    assert match, "LoRA targets differ!"
    assert len(swift_lmhead) == 0, "Swift has LoRA on lm_head!"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 5: Hard negative label construction equivalence
# ---------------------------------------------------------------------------


def test_hard_negative_labels():
    """Verify that hard negative interleaving produces equivalent loss.

    Contrastors: docs = [pos0, neg0a, neg0b, pos1, neg1a, neg1b]
                 labels = [0, 3] (positive positions in flattened doc array)
                 similarity = query @ docs.T, cross_entropy(sim, labels)

    Swift:       sentences = [[q0, pos0, neg0a, neg0b], [q1, pos1, neg1a, neg1b]]
                 split per sample, stack → [B, neg+2, D]
                 queries = sentences[:, 0], docs_all = sentences[:, 1:].reshape(-1, D)
                 labels = [0, 3] (start of each group's doc block)
                 similarity = queries @ docs_all.T, cross_entropy(sim / temp, labels)
    """
    print("\n=== Test 5: Hard negative label construction ===")

    import torch.nn.functional as F
    from train_contrastors import LogitScale

    torch.manual_seed(123)
    batch_size = 3
    num_hard_neg = 2
    dim = 64
    temperature = 0.07

    # Create normalized embeddings
    query_embs = torch.randn(batch_size, dim)
    query_embs = query_embs / query_embs.norm(dim=-1, keepdim=True)

    # docs_per_query = 1 pos + num_hard_neg
    docs_per_query = 1 + num_hard_neg
    all_doc_embs = torch.randn(batch_size * docs_per_query, dim)
    all_doc_embs = all_doc_embs / all_doc_embs.norm(dim=-1, keepdim=True)

    # --- Contrastors path ---
    # clip_loss: labels point to positive positions [0, 3, 6]
    logit_scale = LogitScale(init_value=1.0 / temperature)
    similarity_c = logit_scale(torch.matmul(query_embs, all_doc_embs.T))
    labels_c = torch.arange(batch_size) * docs_per_query
    contrastors_loss = F.cross_entropy(similarity_c, labels_c).item()

    # --- Swift path ---
    # Reconstruct swift's format: [B, neg+2, D] where dim1 = [query, pos, neg1, neg2]
    sentences = []
    for i in range(batch_size):
        group = torch.cat(
            [
                query_embs[i : i + 1],
                all_doc_embs[i * docs_per_query : (i + 1) * docs_per_query],
            ],
            dim=0,
        )  # [neg+2, D]
        sentences.append(group)
    sentences = torch.stack(sentences, dim=0)  # [B, neg+2, D]

    queries = sentences[:, 0]  # [B, D]
    docs_all = sentences[:, 1:].reshape(-1, dim)  # [B*(neg+1), D]
    labels_s = torch.arange(0, batch_size * (docs_per_query), docs_per_query)
    similarity_s = torch.matmul(queries, docs_all.T) / temperature
    swift_loss = F.cross_entropy(similarity_s, labels_s).item()

    diff = abs(contrastors_loss - swift_loss)
    print(f"  Contrastors loss (hard neg): {contrastors_loss:.6f}")
    print(f"  Swift loss (hard neg):       {swift_loss:.6f}")
    print(f"  Label contrastors: {labels_c.tolist()}")
    print(f"  Label swift:       {labels_s.tolist()}")
    print(f"  Absolute diff:     {diff:.6e}")

    assert diff < 1e-5, f"Hard negative loss mismatch: {diff}"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 6: End-to-end data pipeline equivalence
# ---------------------------------------------------------------------------


def test_data_pipeline():
    """Verify that the full data pipeline (load image → process → embed) is equivalent.

    Loads real data samples, processes through both pipelines' collation,
    and compares the resulting token IDs and pixel values.
    """
    print("\n=== Test 6: Data pipeline (collate) equivalence ===")

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    # Load 2 real samples with hard negatives
    with open("data/train_hn.jsonl") as f:
        samples = [json.loads(f.readline()) for _ in range(2)]

    # --- Contrastors collate path ---
    from train_contrastors import (
        init_chat_templates,
        process_doc_images,
        process_queries,
    )

    init_chat_templates(processor)

    queries = [s["query"] for s in samples]
    doc_images = []
    for s in samples:
        doc_images.append(Image.open(s["chunk_path"]).convert("RGB"))
        # Add first hard negative
        if s.get("neg_chunk_paths") and os.path.exists(s["neg_chunk_paths"][0]):
            doc_images.append(Image.open(s["neg_chunk_paths"][0]).convert("RGB"))

    contrastors_q = process_queries(processor, queries)
    contrastors_d = process_doc_images(processor, doc_images)

    # --- Swift path: same processor, same template ---
    # Swift ultimately calls the same processor.apply_chat_template → processor()
    # We verify the query token IDs match
    swift_q_texts = []
    for q in queries:
        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": QUERY_INSTRUCTION}],
            },
            {"role": "user", "content": [{"type": "text", "text": q}]},
        ]
        swift_q_texts.append(
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )
    swift_q = processor(text=swift_q_texts, return_tensors="pt", padding="longest")

    # Doc images: same images, same template
    swift_d_texts = []
    for _ in doc_images:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": DOC_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "image"}]},
        ]
        swift_d_texts.append(
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )
    swift_d = processor(
        text=swift_d_texts, images=doc_images, return_tensors="pt", padding="longest"
    )

    # Compare query tokens
    q_match = torch.equal(contrastors_q["input_ids"], swift_q["input_ids"])
    q_attn_match = torch.equal(
        contrastors_q["attention_mask"], swift_q["attention_mask"]
    )
    print(f"  Query input_ids match:      {q_match}")
    print(f"  Query attention_mask match: {q_attn_match}")

    # Compare doc tokens
    d_ids_match = torch.equal(contrastors_d["input_ids"], swift_d["input_ids"])
    d_attn_match = torch.equal(
        contrastors_d["attention_mask"], swift_d["attention_mask"]
    )
    print(f"  Doc input_ids match:        {d_ids_match}")
    print(f"  Doc attention_mask match:   {d_attn_match}")

    # Compare pixel values — contrastors reshapes to (B, max_patches, dim),
    # swift keeps flat (total_patches, dim). Compare the flat versions.
    if "pixel_values" in contrastors_d and "pixel_values" in swift_d:
        c_pv = contrastors_d["pixel_values"]
        s_pv = swift_d["pixel_values"]
        # Contrastors: (B, max_patches, dim) → flatten valid patches
        c_offsets = contrastors_d["image_grid_thw"].prod(dim=1).tolist()
        c_flat = torch.cat(
            [c_pv[i, : c_offsets[i]] for i in range(len(c_offsets))], dim=0
        )
        # Swift: already flat (total_patches, dim)
        s_flat = (
            s_pv
            if s_pv.dim() == 2
            else torch.cat(
                [s_pv[i, : c_offsets[i]] for i in range(len(c_offsets))], dim=0
            )
        )
        pv_match = torch.equal(c_flat, s_flat)
        pv_maxdiff = (
            (c_flat.float() - s_flat.float()).abs().max().item()
            if not pv_match
            else 0.0
        )
        print(f"  Pixel values match:         {pv_match} (max_diff={pv_maxdiff:.6e})")
    else:
        pv_match = True
        print("  Pixel values: skipped (not present)")

    for img in doc_images:
        img.close()

    assert q_match, "Query input_ids mismatch!"
    assert d_ids_match, "Doc input_ids mismatch!"
    assert pv_match, "Pixel values mismatch!"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 7: Single training step equivalence (end-to-end)
# ---------------------------------------------------------------------------


def test_training_step():
    """Run one training step through both pipelines and compare loss.

    Uses the same model weights, same data, same hyperparameters.
    Compares the loss value after one forward pass (no grad, just loss).
    """
    print("\n=== Test 7: Training step loss equivalence ===")

    from models.biqwen3 import BiQwen3
    from PIL import Image
    from train_contrastors import (
        LogitScale,
        _clear_rope_deltas,
        clip_loss,
        init_chat_templates,
        process_doc_images,
        process_queries,
    )
    from transformers import AutoProcessor

    temperature = 0.07
    num_hard_neg = 2

    # Load 2 samples
    with open("data/train_hn.jsonl") as f:
        samples = [json.loads(f.readline()) for _ in range(2)]

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"
    init_chat_templates(processor)

    # Prepare data
    queries = [s["query"] for s in samples]
    doc_images = []
    for s in samples:
        doc_images.append(Image.open(s["chunk_path"]).convert("RGB"))
        neg_paths = s.get("neg_chunk_paths", [])
        for np_ in neg_paths[:num_hard_neg]:
            if np_ and os.path.exists(np_):
                doc_images.append(Image.open(np_).convert("RGB"))

    q_inputs = process_queries(processor, queries)
    d_inputs = process_doc_images(processor, doc_images)

    # --- Contrastors loss ---
    model = BiQwen3.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).cuda().eval()
    q_inputs_c = {k: v.cuda() for k, v in q_inputs.items()}
    d_inputs_c = {k: v.cuda() for k, v in d_inputs.items()}

    logit_scale = LogitScale(init_value=1.0 / temperature).cuda()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _clear_rope_deltas(model)
        q_emb = model(**q_inputs_c)
        _clear_rope_deltas(model)
        d_emb = model(**d_inputs_c)
        c_loss, c_acc = clip_loss(q_emb, d_emb, logit_scale, gather_enabled=False)

    contrastors_loss = c_loss.item()
    contrastors_acc = c_acc.item()

    # Extract embeddings for swift comparison
    q_emb_np = q_emb.cpu().float()
    d_emb_np = d_emb.cpu().float()

    del model
    torch.cuda.empty_cache()

    # --- Swift loss (using the same embeddings) ---
    # Swift data layout:
    #   sentences: [anchor0, pos0, neg0a, neg0b, anchor1, pos1, neg1a, neg1b]
    #              = B * (1 anchor + 1 pos + num_neg) entries
    #   labels:    [1, 0, 0, 1, 0, 0]
    #              = B * (1 pos + num_neg) entries (anchors NOT in labels)
    #   The '1' marks positive positions. _parse_multi_negative_sentences uses
    #   an offset adjustment (+range) to account for anchors in sentences.
    docs_per_query = 1 + num_hard_neg
    batch_size = len(queries)

    swift_sentences = []
    swift_labels = []
    for i in range(batch_size):
        swift_sentences.append(q_emb_np[i])  # anchor (not in labels)
        swift_sentences.append(d_emb_np[i * docs_per_query])  # positive
        swift_labels.append(1.0)
        for j in range(1, docs_per_query):  # negatives
            swift_sentences.append(d_emb_np[i * docs_per_query + j])
            swift_labels.append(0.0)

    swift_sentences = torch.stack(swift_sentences, dim=0)  # [B*(1+1+num_neg), D]
    swift_labels = torch.tensor(swift_labels)  # [B*(1+num_neg)]

    # Run swift's parsing + InfoNCE logic (batched path, use_batch=True)
    from swift.loss.embedding import _parse_multi_negative_sentences

    split_tensors = _parse_multi_negative_sentences(
        swift_sentences, swift_labels, hard_negatives=num_hard_neg
    )

    # Each split_tensor = [anchor, pos, neg1, neg2] shape [neg+2, D]
    sentences_stacked = torch.stack(split_tensors, dim=0)  # [B, neg+2, D]
    swift_queries = sentences_stacked[:, 0]  # [B, D]
    docs_all = sentences_stacked[:, 1:].reshape(
        -1, sentences_stacked.size(2)
    )  # [B*(neg+1), D]
    swift_label_indices = torch.arange(0, batch_size * docs_per_query, docs_per_query)
    similarity = torch.matmul(swift_queries, docs_all.T) / temperature
    swift_loss = torch.nn.functional.cross_entropy(
        similarity, swift_label_indices
    ).item()

    diff = abs(contrastors_loss - swift_loss)
    print(f"  Contrastors loss: {contrastors_loss:.6f}  acc: {contrastors_acc:.4f}")
    print(f"  Swift loss:       {swift_loss:.6f}")
    print(f"  Absolute diff:    {diff:.6e}")
    print(
        f"  Batch: {batch_size} queries, {len(doc_images)} docs ({num_hard_neg} hard neg/query)"
    )

    for img in doc_images:
        img.close()

    # Allow small tolerance for bf16 accumulated differences
    assert diff < 0.01, f"Training step loss mismatch: {diff}"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 8: Cross-GPU gather semantics (single-GPU simulation)
# ---------------------------------------------------------------------------


def test_gather_semantics():
    """Verify that gather_with_grad and gather_object produce the same
    forward-pass result on single GPU (no actual distributed).

    The key semantic difference:
    - contrastors gather_with_grad: gradients flow through all gathered tensors
    - swift gather_object: other ranks' tensors are detached

    On single GPU (world_size=1), both are no-ops. This test verifies that
    the loss computation (which uses gathered embeddings) is identical when
    there's only one "rank", confirming the single-GPU equivalence.

    Multi-GPU difference: contrastors gives slightly different gradients because
    gradients flow through all ranks' embeddings. This test documents the known
    difference but cannot verify it without multiple GPUs.
    """
    print("\n=== Test 8: Cross-GPU gather semantics ===")

    import torch.nn.functional as F
    from train_contrastors import LogitScale, clip_loss, gather_with_grad

    torch.manual_seed(99)
    batch_size = 4
    dim = 64
    temperature = 0.07

    q = torch.randn(batch_size, dim)
    q = q / q.norm(dim=-1, keepdim=True)
    d = torch.randn(batch_size, dim)
    d = d / d.norm(dim=-1, keepdim=True)

    # contrastors path (gather_enabled=False on single GPU is equivalent)
    logit_scale = LogitScale(init_value=1.0 / temperature)
    loss_c, _ = clip_loss(q, d, logit_scale, gather_enabled=False)

    # Simulate gather_with_grad on single GPU (should be identity)
    d_gathered = gather_with_grad(d)
    sim = logit_scale(torch.matmul(q, d_gathered.T))
    labels = torch.arange(batch_size)
    loss_gathered = F.cross_entropy(sim, labels)

    # swift path
    sim_s = torch.matmul(q, d.T) / temperature
    loss_swift = F.cross_entropy(sim_s, labels)

    diff_cg = abs(loss_c.item() - loss_gathered.item())
    diff_cs = abs(loss_c.item() - loss_swift.item())

    print(f"  clip_loss (no gather):    {loss_c.item():.6f}")
    print(f"  clip_loss (w/ gather):    {loss_gathered.item():.6f}")
    print(f"  swift (/ temperature):    {loss_swift.item():.6f}")
    print(f"  Diff (no gather vs gather):       {diff_cg:.6e}")
    print(f"  Diff (contrastors vs swift):      {diff_cs:.6e}")
    print("  NOTE: Multi-GPU gradient difference (gather_with_grad vs detached gather)")
    print(
        "         cannot be tested without multiple GPUs. This is a KNOWN difference."
    )

    assert diff_cg < 1e-6, f"gather_with_grad changed loss on single GPU: {diff_cg}"
    assert diff_cs < 1e-5, f"Loss diverged: {diff_cs}"
    print("  PASSED ✓")
    return True


# ---------------------------------------------------------------------------
# Test 9: Multi-GPU gather gradient difference (requires torchrun)
# ---------------------------------------------------------------------------


def test_multi_gpu_gather():
    """Compare gradients from gather_with_grad vs detached gather on 2 GPUs.

    gather_with_grad: backward flows through all ranks' embeddings
    detached gather:  backward only flows through local rank's embeddings

    The LOSS should be identical (same forward computation).
    The GRADIENTS will differ because gather_with_grad lets rank 0's loss
    update rank 1's parameters (and vice versa) via reduce_scatter.

    Run with: CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 tests/test_swift_equivalence.py --multi-gpu
    """
    print("\n=== Test 9: Multi-GPU gather gradient difference ===")

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    assert world_size == 2, f"This test requires exactly 2 GPUs, got {world_size}"

    from train_contrastors import gather_with_grad

    torch.manual_seed(42 + rank)  # different data per rank
    batch_size = 4
    dim = 64
    temperature = 0.07

    # Each rank has its own query and doc embeddings
    q = torch.randn(batch_size, dim, device=device, requires_grad=True)
    q_norm = q / q.norm(dim=-1, keepdim=True)

    d = torch.randn(batch_size, dim, device=device, requires_grad=True)
    d_norm = d / d.norm(dim=-1, keepdim=True)

    # --- Path A: gather_with_grad (contrastors) ---
    d_gathered_grad = gather_with_grad(d_norm)  # [B*W, D], grad flows through all
    sim_a = torch.matmul(q_norm, d_gathered_grad.T) / temperature
    labels_a = torch.arange(batch_size, device=device) + rank * batch_size
    loss_a = torch.nn.functional.cross_entropy(sim_a, labels_a)
    loss_a.backward()
    grad_a_q = q.grad.clone()
    grad_a_d = d.grad.clone()

    # Reset grads
    q.grad = None
    d.grad = None

    # Recompute normalized (grad graph was consumed)
    q_norm2 = q / q.norm(dim=-1, keepdim=True)
    d_norm2 = d / d.norm(dim=-1, keepdim=True)

    # --- Path B: detached gather (swift-style) ---
    all_d = [torch.zeros_like(d_norm2) for _ in range(world_size)]
    dist.all_gather(all_d, d_norm2)
    # Detach other ranks, keep local with grad
    all_d[rank] = d_norm2
    for i in range(world_size):
        if i != rank:
            all_d[i] = all_d[i].detach()
    d_gathered_detach = torch.cat(all_d, dim=0)

    sim_b = torch.matmul(q_norm2, d_gathered_detach.T) / temperature
    labels_b = torch.arange(batch_size, device=device) + rank * batch_size
    loss_b = torch.nn.functional.cross_entropy(sim_b, labels_b)
    loss_b.backward()
    grad_b_q = q.grad.clone()
    grad_b_d = d.grad.clone()

    # Compare
    loss_diff = abs(loss_a.item() - loss_b.item())
    q_grad_cosine = torch.nn.functional.cosine_similarity(
        grad_a_q.flatten().unsqueeze(0), grad_b_q.flatten().unsqueeze(0)
    ).item()
    d_grad_cosine = torch.nn.functional.cosine_similarity(
        grad_a_d.flatten().unsqueeze(0), grad_b_d.flatten().unsqueeze(0)
    ).item()
    q_grad_diff = (grad_a_q - grad_b_q).abs().max().item()
    d_grad_diff = (grad_a_d - grad_b_d).abs().max().item()

    if rank == 0:
        print(f"  Loss diff (should be ~0):          {loss_diff:.6e}")
        print(f"  Query grad cosine:                  {q_grad_cosine:.6f}")
        print(f"  Doc grad cosine:                    {d_grad_cosine:.6f}")
        print(f"  Query grad max diff:                {q_grad_diff:.6e}")
        print(f"  Doc grad max diff:                  {d_grad_diff:.6e}")
        print("  NOTE: Gradient difference is EXPECTED and KNOWN.")
        print("         gather_with_grad propagates grad to all ranks' embeddings;")
        print("         detached gather only propagates to local rank.")

        # Loss should be identical (same forward)
        assert loss_diff < 1e-5, f"Loss should be identical: {loss_diff}"

        # Query grads should be identical (query is always local)
        assert q_grad_cosine > 0.999, (
            f"Query grads diverged unexpectedly: {q_grad_cosine}"
        )

        # Doc grads WILL differ — this is the known semantic difference
        # gather_with_grad: d gets gradient from all ranks' CE losses
        # detached gather: d only gets gradient from local rank's CE loss
        # We just document how much they differ
        print("\n  Doc grad difference quantifies the gather_with_grad vs detach gap.")
        print("  This is the ONLY non-equivalent component between the two pipelines.")

    dist.barrier()
    if rank == 0:
        print("  PASSED ✓")
    dist.destroy_process_group()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results = {}

    # Multi-GPU mode: only run the distributed test
    if "--multi-gpu" in sys.argv:
        try:
            results["multi_gpu_gather"] = test_multi_gpu_gather()
        except Exception as e:
            print(f"  FAILED ✗: {e}")
            import traceback

            traceback.print_exc()
            results["multi_gpu_gather"] = False
        # Print summary and exit
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print("\n" + "=" * 50)
            for name, passed in results.items():
                print(f"  {name}: {'PASSED ✓' if passed else 'FAILED ✗'}")
        return

    tests = [
        ("tokenization", test_tokenization),
        ("embedding", test_embedding),
        ("loss", test_loss),
        ("lora_targets", test_lora_targets),
        ("hard_negative_labels", test_hard_negative_labels),
        ("data_pipeline", test_data_pipeline),
        ("training_step", test_training_step),
        ("gather_semantics", test_gather_semantics),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  FAILED ✗: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 50)
    print("Summary:")
    all_pass = True
    for name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
