"""`install_chrome` falls back to the zstandard package whenever zstd can't run.

The fallback caught only FileNotFoundError. Launching a missing zstd doesn't
always raise that: if a directory on PATH can't be searched (WSL can inherit
Windows PATH entries the user has no access to), or a `zstd` there isn't
executable, it raises PermissionError, which escaped as a raw traceback before
the zstandard fallback or the install hint could run.

A non-executable `zstd` alone on PATH raises the same PermissionError on any
POSIX system, even as root. The download is stubbed, so nothing touches the
network or the real install directory.
"""

import io
import json
import subprocess
import sys
import tarfile
import types
import urllib.request

import pytest
from pixelrag_render import chrome

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX PATH lookup; install_chrome is linux-x64 only",
)

_ARCHIVE = b"downloaded headless_shell archive"


def _headless_shell_tar() -> bytes:
    data = b"#!/bin/sh\n"
    member = tarfile.TarInfo("headless_shell")
    member.size = len(data)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.addfile(member, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def install_dir(tmp_path, monkeypatch):
    """A linux-x64 install into tmp_path, with a zstd on PATH that can't run."""
    target = tmp_path / "chrome"
    monkeypatch.setattr(chrome, "INSTALL_DIR", target)
    monkeypatch.setattr(chrome.platform, "system", lambda: "Linux")
    monkeypatch.setattr(chrome.platform, "machine", lambda: "x86_64")

    def download(url, filename, reporthook=None):
        with open(filename, "wb") as f:
            f.write(_ARCHIVE)
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", download)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    zstd = bin_dir / "zstd"
    zstd.write_text("#!/bin/sh\n")
    zstd.chmod(0o644)
    monkeypatch.setenv("PATH", str(bin_dir))
    # The premise: launching zstd now fails with PermissionError, not
    # FileNotFoundError.
    with pytest.raises(PermissionError):
        subprocess.run(["zstd", "--version"], capture_output=True)
    return target


def test_zstd_that_cannot_run_falls_back_to_zstandard(install_dir, monkeypatch):
    read = []

    class ZstdDecompressor:
        def copy_stream(self, f_in, f_out):
            read.append(f_in.read())
            f_out.write(_headless_shell_tar())

    zstandard = types.SimpleNamespace(ZstdDecompressor=ZstdDecompressor)
    monkeypatch.setitem(sys.modules, "zstandard", zstandard)

    binary = chrome.install_chrome(version="0.0.0-test")

    assert read == [_ARCHIVE]  # the fallback decompressed the downloaded archive
    assert binary == install_dir / "headless_shell"
    assert binary.is_file()
    installed = json.loads((install_dir / chrome.VERSION_FILE).read_text())
    assert installed["version"] == "0.0.0-test"


def test_no_decompressor_gives_the_install_hint(install_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "zstandard", None)  # `import zstandard` fails

    with pytest.raises(RuntimeError, match="apt install zstd"):
        chrome.install_chrome(version="0.0.0-test")
