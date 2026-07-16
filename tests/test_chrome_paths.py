"""Cross-platform Chrome resolution candidates (logic is testable on any OS)."""

import os
from pixelrag_render.chrome import _candidate_chrome_paths, find_chrome


def test_macos_candidates_include_chrome_app():
    paths = _candidate_chrome_paths("Darwin")
    assert any("Google Chrome.app/Contents/MacOS/Google Chrome" in p for p in paths)
    # No Linux system paths leak into the macOS candidate list.
    assert "/usr/bin/google-chrome" not in paths


def test_windows_candidates_include_chrome_exe():
    paths = _candidate_chrome_paths("Windows")
    assert any(p.endswith("chrome.exe") and "Google" in p for p in paths)


def test_linux_candidates_include_system_chrome():
    paths = _candidate_chrome_paths("Linux")
    assert "/usr/bin/google-chrome" in paths
    assert "/usr/bin/chromium" in paths


def test_chrome_path_env_is_first(monkeypatch):
    monkeypatch.setenv("CHROME_PATH", "/custom/chrome")
    assert _candidate_chrome_paths("Linux")[0] == "/custom/chrome"


def test_find_chrome_returns_executable():
    # On any supported dev box this resolves to an installed/usable binary.
    path = find_chrome()
    assert os.path.isfile(path) and os.access(path, os.X_OK)


def test_playwright_chromium_sorted_by_revision_not_lexically(monkeypatch):
    """Newer Playwright Chromium revisions must rank ahead of older ones.

    Playwright caches browsers under ``chromium-<revision>`` with a plain
    integer revision. A lexicographic sort ranks ``chromium-999`` above
    ``chromium-1187`` once revisions cross the 3→4 digit boundary, so the
    resolver would pick an *older* browser. Sorting by integer revision fixes
    that — assert the candidate list is newest-first regardless of digit count.
    """
    import glob

    found = [
        "/home/u/.cache/ms-playwright/chromium-999/chrome-linux/chrome",
        "/home/u/.cache/ms-playwright/chromium-1208/chrome-linux/chrome",
        "/home/u/.cache/ms-playwright/chromium-1187/chrome-linux/chrome",
    ]
    monkeypatch.setattr(glob, "glob", lambda pattern: list(found))

    paths = _candidate_chrome_paths("Linux")
    revisions = [p for p in paths if "ms-playwright" in p]
    assert revisions == [
        "/home/u/.cache/ms-playwright/chromium-1208/chrome-linux/chrome",
        "/home/u/.cache/ms-playwright/chromium-1187/chrome-linux/chrome",
        "/home/u/.cache/ms-playwright/chromium-999/chrome-linux/chrome",
    ]
