"""`pixelrag serve --index-dir DIR` finds the files `pixelrag index build` wrote.

`pixelrag index build` writes `articles.json` and `tiles/` inside the index
directory, so serving that directory must not also require them in the current
directory. Explicit flags / env vars still win, and a directory that lacks them
keeps the old `./articles.json` / `./tiles` fallback.
"""

import sys
import types

import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")
from pixelrag_serve import api


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("PIXELRAG_INDEX_DIR", "PIXELRAG_TILES_DIR", "PIXELRAG_ARTICLES_JSON"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def served_args(monkeypatch):
    """Run `main()` with the given argv and return the args handed to `load`."""
    seen = {}
    monkeypatch.setattr(api, "load", lambda args: seen.update(vars(args)))
    monkeypatch.setitem(
        sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None)
    )

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["pixelrag-serve", *argv])
        api.main()
        return types.SimpleNamespace(**seen)

    return run


def test_paths_default_to_the_index_dir(served_args, tmp_path):
    (tmp_path / "articles.json").write_text("[]")
    (tmp_path / "tiles").mkdir()

    args = served_args("--index-dir", str(tmp_path))

    assert args.articles_json == str(tmp_path / "articles.json")
    assert args.tiles_dir == str(tmp_path / "tiles")


def test_falls_back_to_the_working_directory(served_args, tmp_path):
    args = served_args("--index-dir", str(tmp_path))

    assert args.articles_json == "./articles.json"
    assert args.tiles_dir == "./tiles"


def test_explicit_flags_win(served_args, tmp_path):
    (tmp_path / "articles.json").write_text("[]")
    (tmp_path / "tiles").mkdir()

    args = served_args(
        "--index-dir",
        str(tmp_path),
        "--articles-json",
        "/data/articles.json",
        "--tiles-dir",
        "/data/tiles",
    )

    assert args.articles_json == "/data/articles.json"
    assert args.tiles_dir == "/data/tiles"


def test_env_vars_win(served_args, tmp_path, monkeypatch):
    (tmp_path / "articles.json").write_text("[]")
    (tmp_path / "tiles").mkdir()
    monkeypatch.setenv("PIXELRAG_ARTICLES_JSON", "/data/articles.json")
    monkeypatch.setenv("PIXELRAG_TILES_DIR", "/data/tiles")

    args = served_args("--index-dir", str(tmp_path))

    assert args.articles_json == "/data/articles.json"
    assert args.tiles_dir == "/data/tiles"
