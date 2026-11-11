#!/usr/bin/env python3
"""Evaluation: Recall@K and MRR for query-chunk retrieval."""

import json
import logging

import faiss
import numpy as np
import torch
from PIL import Image
from training.dataset import DOC_INSTRUCTION, QUERY_INSTRUCTION
from training.model import pool_and_normalize

logger = logging.getLogger(__name__)


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_eval(
    model,
    processor,
    eval_jsonl: str,
    device: str,
    batch_size: int = 16,
    max_pairs: int = 200,
) -> tuple[float, float, float]:
    """Embed eval queries + images, compute Recall@1, Recall@10, MRR.

    Args:
        max_pairs: Cap eval set size for speed. Use 0 for no limit.
    """
    model.eval()

    pairs = []
    with open(eval_jsonl) as f:
        for line in f:
            item = json.loads(line)
            pairs.append((item["query"], item["chunk_path"]))
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    q_embs_list = []
    i_embs_list = []

    with torch.no_grad():
        for batch_pairs in _chunk(pairs, batch_size):
            # Filter out bad images
            valid = []
            for query, path in batch_pairs:
                try:
                    img = Image.open(path).convert("RGB")
                    valid.append((query, img))
                except Exception as e:
                    logger.warning(f"Eval: skipping bad image {path}: {e}")
            if not valid:
                continue

            queries, images = zip(*valid)

            # Query embeddings
            q_messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": QUERY_INSTRUCTION}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": q}]},
                ]
                for q in queries
            ]
            q_texts = [
                processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in q_messages
            ]
            q_inputs = processor(text=q_texts, return_tensors="pt", padding=True)
            q_inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in q_inputs.items()
            }
            q_out = model(**q_inputs, output_hidden_states=True)
            q_emb = pool_and_normalize(
                q_out.hidden_states[-1], q_inputs["attention_mask"]
            )
            q_embs_list.append(q_emb.cpu().float().numpy())

            # Image embeddings
            i_messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": DOC_INSTRUCTION}],
                    },
                    {"role": "user", "content": [{"type": "image", "image": img}]},
                ]
                for img in images
            ]
            i_texts = [
                processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in i_messages
            ]
            i_inputs = processor(
                text=i_texts,
                images=list(images),
                return_tensors="pt",
                padding=True,
                device=device,
            )
            i_inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in i_inputs.items()
            }
            i_out = model(**i_inputs, output_hidden_states=True)
            i_emb = pool_and_normalize(
                i_out.hidden_states[-1], i_inputs["attention_mask"]
            )
            i_embs_list.append(i_emb.cpu().float().numpy())

    if not q_embs_list:
        logger.warning("No valid eval pairs found")
        return 0.0, 0.0, 0.0

    q_embs = np.vstack(q_embs_list).astype(np.float32)
    i_embs = np.vstack(i_embs_list).astype(np.float32)

    n = q_embs.shape[0]
    d = q_embs.shape[1]

    # FAISS inner-product search (embeddings are L2-normalized → IP = cosine)
    index = faiss.IndexFlatIP(d)
    index.add(i_embs)
    k = min(100, n)
    _, indices = index.search(q_embs, k)

    # Each query's correct match is at the same index (diagonal)
    recall_1 = 0.0
    recall_10 = 0.0
    mrr = 0.0
    for i in range(n):
        retrieved = indices[i].tolist()
        if i in retrieved[:1]:
            recall_1 += 1
        if i in retrieved[:10]:
            recall_10 += 1
        if i in retrieved:
            rank = retrieved.index(i) + 1
            mrr += 1.0 / rank

    recall_1 /= n
    recall_10 /= n
    mrr /= n

    model.train()
    return recall_1, recall_10, mrr
