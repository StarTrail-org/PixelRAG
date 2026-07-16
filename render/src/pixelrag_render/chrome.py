"""Chrome binary management for pixelshot.

Downloads and manages a patched headless Chrome binary with rawFilePath
support. Similar to `playwright install chromium`.

Usage:
    pixelshot install-chrome     # download patched headless_shell
    pixelshot which-chrome       # print path to active binary

Programmatic:
    from pixelrag_render.chrome import find_chrome, install_chrome
    path = find_chrome()             # auto-detect best available
    path = install_chrome()          # download if needed
"""

import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

INSTALL_DIR = Path.home() / ".cache" / "pixelrag" / "chrome"
VERSION_FILE = "version.json"

# Update these when releasing a new build
CHROME_VERSION = "150.0.7844.0"
RELEASE_URL_TEMPLATE = (
    "https://github.com/StarTrail-org/PixelRAG/releases/download/"
    "chrome-{version}/headless_shell-linux-x64.tar.zst"
)


def _playwright_revision(path: str) -> int:
    """Integer Playwright Chromium revision embedded in ``path``.

    Playwright caches browsers under ``chromium-<revision>`` with a plain
    monotonically-increasing integer revision. Paths without a recognizable
    revision sort last (``-1``) rather than raising.
    """
    m = re.search(r"chromium-(\d+)", path)
    return int(m.group(1)) if m else -1


def _candidate_chrome_paths(system: str | None = None) -> list[str]:
    """Ordered Chrome binary candidates for the given OS (default: this OS).

    Order: CHROME_PATH env → pixelrag-installed patched headless_shell →
    Playwright's Chromium (newest version first) → system Chrome/Chromium.
    Playwright and system locations are OS-specific so the skill works on
    macOS and Windows, not only Linux.
    """
    import glob

    system = system or platform.system()
    home = Path.home()
    paths: list[str] = []

    env = os.environ.get("CHROME_PATH", "")
    if env:
        paths.append(env)
    # Bundled patched headless_shell (only installed on linux-x64, harmless elsewhere).
    paths.append(str(INSTALL_DIR / "headless_shell"))

    def add_playwright(cache_dir: Path, rel_glob: str) -> None:
        # Newest chromium-NNNN first. Sort by integer revision, not lexically:
        # a string sort ranks "chromium-999" above "chromium-1187" once the
        # revision crosses the 3→4 digit boundary, picking an older browser.
        matches = glob.glob(str(cache_dir / rel_glob))
        paths.extend(sorted(matches, key=_playwright_revision, reverse=True))

    if system == "Darwin":
        add_playwright(
            home / "Library" / "Caches" / "ms-playwright",
            "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        )
        paths += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    elif system == "Windows":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            add_playwright(
                Path(localappdata) / "ms-playwright",
                "chromium-*/chrome-win*/chrome.exe",
            )
        for base in (
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            localappdata,
        ):
            if base:
                paths.append(str(Path(base) / "Google/Chrome/Application/chrome.exe"))
    else:  # Linux / other
        add_playwright(
            home / ".cache" / "ms-playwright", "chromium-*/chrome-linux*/chrome"
        )
        paths += [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
        ]
    return paths


def find_chrome(auto_install: bool = True) -> str:
    """Find the best available Chrome binary. Auto-installs on linux-x64 if none found.

    Search order (per OS): CHROME_PATH → pixelrag-installed headless_shell →
    Playwright's Chromium → system Chrome/Chromium → (linux-x64) auto-install.

    Returns:
        Path to Chrome binary.

    Raises:
        FileNotFoundError: No Chrome binary found (and auto-install unavailable).
    """
    for path in _candidate_chrome_paths():
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # The prebuilt (turbo) headless_shell is published only for linux-x64.
    if auto_install and platform.system() == "Linux" and platform.machine() == "x86_64":
        print("No Chrome found. Installing headless_shell...", flush=True)
        return str(install_chrome())

    raise FileNotFoundError(
        "No Chrome binary found. Install Google Chrome or Chromium, or set CHROME_PATH "
        "to its executable. (The bundled headless_shell auto-installs on linux-x64 only.)"
    )


def get_installed_version() -> str | None:
    """Return version string of installed headless_shell, or None."""
    version_path = INSTALL_DIR / VERSION_FILE
    if version_path.exists():
        try:
            data = json.loads(version_path.read_text())
            return data.get("version")
        except Exception:
            pass
    return None


def is_turbo_capable(chrome_path: str) -> bool:
    """Whether ``chrome_path`` is the pixelrag-installed patched headless_shell,
    which supports the turbo capture extensions (rawFilePath/directClip/skipRedraw).

    Deterministic by provenance — the patched binary lives only at the install path
    (with its version.json marker). No runtime probe, so a stock Chrome is never
    mistaken for a turbo-capable one (and a turbo run is never tried on a binary
    that would hang on the unknown CDP params).
    """
    try:
        installed = (INSTALL_DIR / "headless_shell").resolve()
        return (
            Path(chrome_path).resolve() == installed
            and (INSTALL_DIR / VERSION_FILE).exists()
        )
    except Exception:
        return False


def install_chrome(version: str | None = None, force: bool = False) -> Path:
    """Download and install the patched headless_shell binary.

    Args:
        version: Chrome version to install. Defaults to CHROME_VERSION.
        force: Re-download even if already installed.

    Returns:
        Path to the installed headless_shell binary.
    """
    version = version or CHROME_VERSION
    binary_path = INSTALL_DIR / "headless_shell"

    if binary_path.exists() and not force:
        installed = get_installed_version()
        if installed == version:
            print(f"Already installed: headless_shell {version}")
            return binary_path

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            f"Pre-built headless_shell only available for linux-x64, "
            f"got {platform.system()}-{platform.machine()}"
        )

    url = RELEASE_URL_TEMPLATE.format(version=version)
    print(f"Downloading headless_shell {version}...")
    print(f"  URL: {url}")

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path, _progress_hook)
        print()

        # Decompress: zstd → tar → extract
        print("Extracting...")
        # Try zstd decompression
        decomp_path = tmp_path + ".tar"
        try:
            subprocess.run(
                ["zstd", "-d", tmp_path, "-o", decomp_path],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fallback: try python zstandard
            try:
                import zstandard

                with open(tmp_path, "rb") as f_in, open(decomp_path, "wb") as f_out:
                    dctx = zstandard.ZstdDecompressor()
                    dctx.copy_stream(f_in, f_out)
            except ImportError:
                raise RuntimeError(
                    "zstd not found. Install with: apt install zstd (or pip install zstandard)"
                )

        with tarfile.open(decomp_path) as tar:
            tar.extractall(INSTALL_DIR)
        os.unlink(decomp_path)

        # Set executable permission
        binary_path.chmod(0o755)

        # Write version file
        version_data = {"version": version, "binary": str(binary_path)}
        (INSTALL_DIR / VERSION_FILE).write_text(json.dumps(version_data))

        print(
            f"Installed: {binary_path} ({binary_path.stat().st_size / 1024 / 1024:.0f}MB)"
        )
        return binary_path

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)


def main():
    """CLI entry point for chrome management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage Chrome for pixelshot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("install", help="Download patched headless_shell")
    sub.add_parser("which", help="Print path to active Chrome binary")
    sub.add_parser("version", help="Print installed version")

    args = parser.parse_args()

    if args.command == "install":
        install_chrome()
    elif args.command == "which":
        try:
            print(find_chrome())
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    elif args.command == "version":
        v = get_installed_version()
        if v:
            print(v)
        else:
            print("Not installed", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
