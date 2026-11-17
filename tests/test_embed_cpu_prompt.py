"""embed_cpu must embed a tile the way the GPU embedder does.

`pixelrag index build` routes `embed.device` cpu, mps and auto (the README's
setting) to embed_cpu, and serve scores every index with a query encoder that
matches the GPU embedder. Where a tile lands depends on the prompt around the
image and on which hidden state is pooled. embed_cpu kept its own copy of
both -- an extra "What is shown in this image?" user turn, and the full
model's hidden_states[-1], which is taken before the final norm -- so its
vectors sat far from the GPU path's for the same tile.

The processor and model are fakes: no weights, no network.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

# torch and transformers ship in the optional `embed` extra; CI syncs only
# `--extra dev`, which is torch-free on purpose. Skip rather than fail, as the
# suite does for the other extras (see test_pdf_manifest.py).
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pixelrag_embed import embed_cpu
from pixelrag_embed.embed import DEFAULT_INSTRUCTION, _build_chat_prompt

HIDDEN_SIZE = 8
SEQ_LEN = 5


class FakeProcessor:
    """Renders a conversation deterministically and records what gets embedded."""

    def __init__(self):
        self.texts = []

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        turns = [
            [turn["role"], [part.get("text", part["type"]) for part in turn["content"]]]
            for turn in conversation
        ]
        return json.dumps([turns, add_generation_prompt])

    def __call__(self, text, images, return_tensors, padding):
        self.texts.extend(text)
        ones = torch.ones(1, SEQ_LEN, dtype=torch.long)
        return {"input_ids": ones, "attention_mask": ones}


class FakeModel:
    """The surface of Qwen3VLForConditionalGeneration that embed_cpu touches.

    `model.model(...)` is the base model, whose last_hidden_state is the
    post-norm state serve pools. Calling the full model returns hidden_states
    whose last entry is a different tensor, as on the real model, where it is
    the pre-norm state.
    """

    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        self.last_hidden_state = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, generator=gen)
        self.pre_norm_state = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, generator=gen)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=HIDDEN_SIZE)
        )
        self.full_forward_calls = 0

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def model(self, **inputs):
        return SimpleNamespace(last_hidden_state=self.last_hidden_state)

    def __call__(self, **inputs):
        self.full_forward_calls += 1
        return SimpleNamespace(hidden_states=(self.pre_norm_state,))


@pytest.fixture
def fakes(monkeypatch):
    processor, model = FakeProcessor(), FakeModel()
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: processor
    )
    monkeypatch.setattr(
        transformers.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        lambda *a, **kw: model,
    )
    return processor, model


def _write_tiles(directory, count=2):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = directory / f"chunk_{i:04d}.png"
        Image.new("RGB", (32, 48), "white").save(path)
        paths.append(path)
    return paths


def test_documents_are_embedded_with_the_gpu_embedder_prompt(
    fakes, tmp_path, monkeypatch
):
    """The CLI that `pixelrag index build` runs must use the GPU embedder's prompt.

    No --instruction, as in the pipeline: the default has to be the GPU
    embedder's instruction, not an empty system turn.
    """
    processor, _ = fakes
    shard = tmp_path / "tiles"
    tile_dir = shard / "0.png.tiles"
    tiles = _write_tiles(tile_dir)
    chunks = [{"file": path.name, "chunk_index": i} for i, path in enumerate(tiles)]
    (tile_dir / "chunks.json").write_text(
        json.dumps({"article_id": 0, "chunks": chunks})
    )
    out = tmp_path / "embeddings"
    args = ["--shard-dir", str(shard), "--output-dir", str(out), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", ["embed_cpu", *args])

    embed_cpu.main()

    expected = _build_chat_prompt(processor, DEFAULT_INSTRUCTION)
    assert processor.texts == [expected, expected]
    with np.load(out / "shard_000.npz") as saved:
        assert saved["embeddings"].shape == (2, HIDDEN_SIZE)


def test_instruction_sets_the_system_prompt(fakes, tmp_path):
    """An explicit instruction replaces the system turn, as on the GPU path."""
    processor, _ = fakes
    items = [{"path": str(path)} for path in _write_tiles(tmp_path)]

    embed_cpu.embed_items(
        items, "fake/model", device="cpu", instruction="Represent this page."
    )

    expected = _build_chat_prompt(processor, "Represent this page.")
    assert processor.texts == [expected, expected]


def test_embeddings_pool_the_base_models_last_hidden_state(fakes, tmp_path):
    """Vectors come from the post-norm state; the LM-head forward never runs."""
    _, model = fakes
    items = [{"path": str(path)} for path in _write_tiles(tmp_path)]

    vectors = embed_cpu.embed_items(items, "fake/model", device="cpu")

    last = model.last_hidden_state[0, -1]
    expected = (last / last.norm()).numpy()
    np.testing.assert_allclose(vectors, np.stack([expected, expected]), atol=1e-3)
    assert model.full_forward_calls == 0
