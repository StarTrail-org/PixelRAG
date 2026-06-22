"""Chrome connection implementations: websocket and Playwright."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import urllib.request

CHROME_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-gpu-rasterization",
    "--force-gpu-rasterization",
]


def pick_page_ws_url(targets: list[dict]) -> str:
    """Return the webSocketDebuggerUrl for a ``type: page`` target.

    Chrome's ``/json`` endpoint may list ``background_page`` targets (from
    built-in extensions like Cast / Media Router) before real page targets.
    Connecting to a ``background_page`` hangs because it ignores
    ``Page.navigate``.  Filter for ``type == "page"`` first; fall back to
    the unfiltered list so pre-existing behaviour is preserved when Chrome
    reports no page targets at all.
    """
    pages = [t for t in targets if t.get("type") == "page"] or targets
    return pages[0]["webSocketDebuggerUrl"]


class WebsocketConnection:
    """Direct websocket CDP connection."""

    def __init__(self, ws, proc):
        self._ws = ws
        self._proc = proc
        self._msg_id = 0
        # Pending response futures keyed by message id.
        self._pending: dict[int, asyncio.Future] = {}
        # Listeners for CDP events keyed by method name; each value is a list
        # of (Future, filter_fn) pairs.  filter_fn receives the event params
        # dict and should return True to resolve the future.
        self._event_listeners: dict[str, list] = {}
        self._recv_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Internal receive loop — started lazily on first use.
    # ------------------------------------------------------------------

    def _ensure_recv_loop(self):
        if self._recv_task is None or self._recv_task.done():
            loop = asyncio.get_event_loop()
            self._recv_task = loop.create_task(self._recv_loop())

    async def _recv_loop(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_id = msg.get("id")
                if msg_id is not None:
                    # Response to a command.
                    fut = self._pending.pop(msg_id, None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    # Event notification.
                    method = msg.get("method", "")
                    listeners = self._event_listeners.get(method, [])
                    remaining = []
                    for fut, filter_fn in listeners:
                        if fut.done():
                            continue
                        params = msg.get("params", {})
                        try:
                            matched = filter_fn(params) if filter_fn else True
                        except Exception:
                            matched = True
                        if matched:
                            fut.set_result(params)
                        else:
                            remaining.append((fut, filter_fn))
                    if remaining:
                        self._event_listeners[method] = remaining
                    elif listeners:
                        # All listeners matched or were done — clean up
                        # the stale list so it doesn't accumulate entries.
                        self._event_listeners.pop(method, None)
        except Exception:
            # Socket closed or error — resolve all pending futures with an
            # exception so callers don't hang.
            exc = ConnectionError("WebSocket receive loop ended")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)
            for listeners in self._event_listeners.values():
                for fut, _ in listeners:
                    if not fut.done():
                        fut.set_exception(exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def cdp(self, method: str, params: dict | None = None) -> dict:
        self._ensure_recv_loop()
        self._msg_id += 1
        mid = self._msg_id
        msg = {"id": mid, "method": method}
        if params:
            msg["params"] = params
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps(msg))
        return await asyncio.wait_for(fut, timeout=180)

    async def wait_for_event(
        self,
        method: str,
        timeout: float = 30.0,
        filter_fn=None,
    ) -> dict:
        """Wait for a CDP event with the given method name.

        Args:
            method: CDP event method, e.g. "Page.frameStoppedLoading".
                    NOTE: Prefer Page.frameStoppedLoading over
                    Page.frameNavigated for navigation waits — Chrome has a
                    bug with --in-process-gpu where Page.frameNavigated is
                    sometimes never fired when many instances navigate
                    concurrently.
            timeout: Seconds to wait before raising asyncio.TimeoutError.
            filter_fn: Optional callable(params) -> bool.  The future is
                       resolved only when filter_fn returns True.  If None,
                       the first matching event resolves the future.

        Returns:
            The event ``params`` dict.
        """
        self._ensure_recv_loop()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._event_listeners.setdefault(method, []).append((fut, filter_fn))
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            # Remove the stale listener.
            listeners = self._event_listeners.get(method, [])
            self._event_listeners[method] = [
                (f, fn) for f, fn in listeners if f is not fut
            ]
            if not fut.done():
                fut.cancel()
            raise

    async def close(self):
        try:
            await self._ws.close()
        except Exception:
            pass
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class PlaywrightConnection:
    """Playwright-managed CDP connection."""

    def __init__(self, page, cdp_session, browser, pw):
        self._page = page
        self._cdp = cdp_session
        self._browser = browser
        self._pw = pw

    async def cdp(self, method: str, params: dict | None = None) -> dict:
        result = await self._cdp.send(method, params or {})
        return {"result": result}

    async def close(self):
        try:
            await self._browser.close()
        except Exception:
            pass
        try:
            await self._pw.stop()
        except Exception:
            pass


async def launch_websocket(
    chrome_path: str,
    port: int,
    headless_shell: bool = False,
    extra_args: list[str] | None = None,
) -> WebsocketConnection:
    """Launch Chrome via subprocess, connect via websocket."""
    import websockets

    args = [chrome_path, f"--remote-debugging-port={port}"]
    if not headless_shell:
        args.append("--headless=new")
    args += CHROME_ARGS
    if extra_args:
        args += extra_args
    args += ["about:blank"]

    # Reserve last 8 cores for compression workers
    n_cpus = os.cpu_count() or 128
    chrome_cores = list(range(max(1, n_cpus - 8)))

    def _preexec():
        try:
            os.sched_setaffinity(0, chrome_cores)
        except OSError:
            pass

    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=_preexec,
    )

    for attempt in range(10):
        await asyncio.sleep(1)
        try:
            data = urllib.request.urlopen(
                f"http://localhost:{port}/json", timeout=3
            ).read()
            targets = json.loads(data)
            ws = await websockets.connect(
                pick_page_ws_url(targets),
                open_timeout=10,
                max_size=50 * 1024 * 1024,
            )
            return WebsocketConnection(ws, proc)
        except Exception:
            if attempt == 9:
                proc.kill()
                raise ConnectionError(f"Failed to connect to Chrome on port {port}")


async def launch_playwright(chrome_path: str) -> PlaywrightConnection:
    """Launch Chrome via Playwright, get CDP session."""
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True, executable_path=chrome_path, args=CHROME_ARGS
    )
    ctx = await browser.new_context(
        viewport={"width": 875, "height": 8192},
        device_scale_factor=1,
    )
    page = await ctx.new_page()
    cdp_session = await ctx.new_cdp_session(page)
    return PlaywrightConnection(page, cdp_session, browser, pw)
