"""A Chrome that dies at launch must be reported, not swallowed.

When the browser process exits before CDP is reachable (e.g. Chromium's
"No usable sandbox" abort on Ubuntu 24.04), `pixelshot` used to log
`done=0 failed=0` and nothing else: the worker's exception was dropped by
`asyncio.gather(return_exceptions=True)` and its URLs never counted as failed.

These run on a core install: the "browser" is a shell stub that prints a Chrome
error and exits, so nothing is ever rendered.
"""

import asyncio
import logging
import subprocess
import sys

import pytest
from pixelrag_render.backends import cdp

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="uses a POSIX shell script as the browser"
)


@pytest.fixture
def dead_chrome(tmp_path, monkeypatch):
    """A `chrome` that aborts at startup, with the sandbox error from #194."""
    stub = tmp_path / "chrome"
    stub.write_text(
        "#!/bin/sh\n"
        "echo 'FATAL:zygote_host_impl_linux.cc:132] No usable sandbox!' >&2\n"
        "exit 1\n"
    )
    stub.chmod(0o755)

    # Skip the real waits: the 3s launch sleep and the connect retries.
    real_sleep = asyncio.sleep

    async def no_wait(_seconds, *args, **kwargs):
        await real_sleep(0)

    launched = []
    real_popen = subprocess.Popen

    def spy_popen(*args, **kwargs):
        launched.append(real_popen(*args, **kwargs))
        return launched[-1]

    async def refuse(port, *args, **kwargs):
        # By the time the real retries run out, Chrome has long since died.
        for proc in launched:
            proc.wait(timeout=30)
        raise ConnectionError(f"Failed to connect to Chrome on port {port}")

    monkeypatch.setattr(cdp.subprocess, "Popen", spy_popen)
    monkeypatch.setattr(cdp.asyncio, "sleep", no_wait)
    monkeypatch.setattr(cdp, "_connect_cdp", refuse)
    monkeypatch.delenv("PIXELSHOT_CDP_URL", raising=False)
    return stub


def _render(stub, out, urls):
    return cdp.render_urls(urls, out, chrome_path=str(stub), turbo=False, workers=1)


def test_chrome_exit_and_stderr_are_logged(dead_chrome, tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger=cdp.logger.name):
        results = _render(dead_chrome, tmp_path / "out", ["https://example.com"])

    assert results == []
    errors = "\n".join(r.getMessage() for r in caplog.records)
    assert "Failed to connect to Chrome" in errors
    assert "exited with code 1" in errors
    assert "No usable sandbox" in errors


def test_unrendered_urls_count_as_failed(dead_chrome, tmp_path, caplog):
    urls = ["https://example.com/a", "https://example.com/b"]

    with caplog.at_level(logging.INFO, logger=cdp.logger.name):
        _render(dead_chrome, tmp_path / "out", urls)

    messages = [r.getMessage() for r in caplog.records]
    assert "Batch complete: done=0 failed=2" in messages


def test_a_live_chrome_is_not_blamed_for_a_failed_connect():
    class Running:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("chrome", timeout)

    assert cdp._chrome_exit_detail(Running(), "/nonexistent") == ""
