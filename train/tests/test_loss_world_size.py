"""Data-parallel gradients must match the single-process gradient on the global batch.

Runs grad_cache_loss on 2 CPU ranks (gloo) with a tiny model and compares each rank's
synced gradients with one process computing the same loss on the whole global batch.
No GPU needed:

    pytest train/tests/test_loss_world_size.py -v
"""

import functools
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

TRAIN_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, TRAIN_DIR)

import train_contrastors as tc  # noqa: E402

WORLD = 2
QUERIES_PER_RANK = 4
CHUNK = 2
DIM_IN, DIM_OUT = 12, 8

CASES = {
    "clip": (tc.clip_loss, 0),
    "clip_hard_negatives": (tc.clip_loss, 1),
    "clip_label_smoothing": (functools.partial(tc.clip_loss, label_smoothing=0.1), 1),
    "siglip": (tc.siglip_loss, 1),
}


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, 16), torch.nn.Tanh(), torch.nn.Linear(16, DIM_OUT)
        )

    def forward(self, input_ids, bidirectional=False):
        return F.normalize(self.net(input_ids), dim=-1)


def make_logit_scale():
    scale = tc.LogitScale(init_value=1 / 0.07)
    with torch.no_grad():
        scale.log_scale.fill_(2.5)
    return scale


def global_batch(hard_negatives):
    """Queries and documents for all ranks; each query owns 1 + hard_negatives docs."""
    g = torch.Generator().manual_seed(1)
    n = WORLD * QUERIES_PER_RANK
    q = torch.randn(n, DIM_IN, generator=g)
    d = torch.randn(n * (1 + hard_negatives), DIM_IN, generator=g)
    d.view(n, 1 + hard_negatives, DIM_IN)[:, 0] += 1.5 * q
    return q, d


def grads(model, scale):
    out = [p.grad.detach().clone() for p in model.parameters()]
    out.append(scale.log_scale.grad.detach().clone())
    return out


def _worker(rank, port, case, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    loss_fn, hard_negatives = CASES[case]
    q, d = global_batch(hard_negatives)
    per_doc = QUERIES_PER_RANK * (1 + hard_negatives)
    q = q[rank * QUERIES_PER_RANK:(rank + 1) * QUERIES_PER_RANK]
    d = d[rank * per_doc:(rank + 1) * per_doc]

    model = DDP(Tower(), broadcast_buffers=False)
    scale = make_logit_scale()
    tc.grad_cache_loss(
        model=model,
        query_chunks=[{"input_ids": t} for t in q.split(CHUNK)],
        doc_chunks=[{"input_ids": t} for t in d.split(CHUNK)],
        logit_scale=scale,
        query_process_fn=tc.forward_query,
        doc_process_fn=tc.forward_doc,
        loss_fn=loss_fn,
    )
    torch.save(grads(model.module, scale), os.path.join(out_dir, f"rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("case", list(CASES))
def test_data_parallel_grads_match_global_batch(case, tmp_path):
    mp.start_processes(
        _worker, args=(free_port(), case, str(tmp_path)), nprocs=WORLD, start_method="spawn"
    )

    loss_fn, hard_negatives = CASES[case]
    q, d = global_batch(hard_negatives)
    model, scale = Tower(), make_logit_scale()
    loss, _ = loss_fn(model(q), model(d), scale, gather_enabled=False)
    loss.backward()
    expected = grads(model, scale)

    for rank in range(WORLD):
        got = torch.load(tmp_path / f"rank{rank}.pt")
        for i, (g, e) in enumerate(zip(got, expected)):
            ratio = (g.norm() / e.norm()).item()
            assert torch.allclose(g, e, rtol=1e-4, atol=1e-6), (
                f"rank {rank}, parameter {i}: gradient is {ratio:.3f}x the "
                f"global-batch gradient"
            )
