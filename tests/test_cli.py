"""Smoke tests for the CLI entry points and light imports.

These run on a core `pip install pixelrag` (no torch, no chrome): the `pixelshot`
and `pixelrag` commands must work and the light modules must import.
"""

import subprocess
import sys
from pathlib import Path

# Console scripts live next to the interpreter running the tests (works whether
# invoked via `uv run pytest` or `.venv/bin/python -m pytest`).
_BIN = Path(sys.executable).parent


def _run(script, *args):
    return subprocess.run([str(_BIN / script), *args], capture_output=True, text=True)


def test_pixelshot_help():
    r = _run("pixelshot", "--help")
    assert r.returncode == 0
    assert "pixelshot" in r.stdout


def test_pixelrag_umbrella_help():
    r = _run("pixelrag", "--help")
    assert r.returncode == 0
    out = r.stdout.lower()
    assert "stage" in out
    for stage in ("chunk", "embed", "build-index", "index", "serve"):
        assert stage in out


def test_index_build_device_choices():
    # CLI must offer every device the embedder supports (auto/mps were missing).
    # Assert the argparse-rendered choices token, not loose words in the help prose.
    r = _run("pixelrag", "index", "build", "--help")
    assert r.returncode == 0
    assert "{auto,cpu,mps,cuda}" in r.stdout


def test_pixelrag_unknown_stage_errors():
    r = _run("pixelrag", "definitely-not-a-stage")
    assert r.returncode != 0
    assert "unknown" in (r.stdout + r.stderr).lower()


def test_build_index_help_lists_qdrant_collection_modes():
    r = _run("pixelrag", "build-index", "build", "--help")
    assert r.returncode == 0
    assert "--append" in r.stdout
    assert "--recreate" in r.stdout
    assert "--qdrant-client-config" in r.stdout
    assert "--qdrant-quantization-config" in r.stdout


def test_light_imports():
    # Core install must import without torch.
    import pixelrag  # noqa: F401
    import pixelrag_render  # noqa: F401
    from pixelrag_render import render_file, render_url  # noqa: F401


def test_dispatcher_stage_table():
    from pixelrag.cli import STAGES

    assert set(STAGES) == {
        "chunk",
        "embed",
        "build-index",
        "index",
        "monitor",
        "serve",
    }
