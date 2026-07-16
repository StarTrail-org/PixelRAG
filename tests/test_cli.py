"""Smoke tests for the CLI entry points and light imports.

These run on a core `pip install pixelrag` (no torch, no chrome): the `pixelshot`
and `pixelrag` commands must work and the light modules must import.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Console scripts live next to the interpreter running the tests (works whether
# invoked via `uv run pytest` or `.venv/bin/python -m pytest`).
_BIN = Path(sys.executable).parent


def _run(script, *args):
    return subprocess.run([str(_BIN / script), *args], capture_output=True, text=True)


def test_pixelshot_help():
    r = _run("pixelshot", "--help")
    assert r.returncode == 0
    assert "pixelshot" in r.stdout


def test_pixelshot_txt_input_uses_batch_render(monkeypatch, tmp_path, capsys):
    from pixelrag_render import render as render_mod

    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "\n https://example.com/a \n\nhttps://example.com/b\nhttps://example.com/c\n"
    )
    calls = []

    def fake_render_urls(urls, output_dir, **kwargs):
        calls.append((list(urls), kwargs["workers"]))
        return [Path(output_dir) / "a.png.tiles", Path(output_dir) / "c.png.tiles"]

    monkeypatch.setattr(render_mod, "render_urls", fake_render_urls)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pixelshot", str(urls_file), "-o", str(tmp_path / "out"), "-w", "8"],
    )

    render_mod.main()

    assert calls == [
        (
            [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            8,
        )
    ]
    stdout = capsys.readouterr().out
    assert "a.png.tiles" in stdout
    assert "b.png.tiles" not in stdout
    assert "c.png.tiles" in stdout


def test_pixelshot_missing_txt_input(monkeypatch, tmp_path, capsys):
    from pixelrag_render import render as render_mod

    missing = tmp_path / "missing.txt"
    monkeypatch.setattr(sys, "argv", ["pixelshot", str(missing)])

    with pytest.raises(SystemExit) as exc:
        render_mod.main()

    assert exc.value.code == 2
    assert "URL file not found" in capsys.readouterr().err


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
