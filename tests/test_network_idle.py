"""networkidle2 readiness: in-flight tolerance, load gating, and the hard cap.

The old in-page probe reset its quiet timer on every resource *completion*,
so any page with sub-500ms analytics beacons rode the 12s hard cap. The
Python-side watcher counts *in-flight* requests instead (Puppeteer's
networkidle2 semantics): <=2 pending requests count as quiet, persistent
connections never block, and >2 permanently-busy connections still exit at
the cap rather than hanging.
"""

import asyncio
import json
import time

from pixelrag_render.backends.cdp import (
    _NetIdleState,
    _wait_load_and_network_idle,
)


def _open(state, rid, now):
    state.on_frame(
        {"method": "Network.requestWillBeSent", "params": {"requestId": rid}}, now
    )


def _close(state, rid, now):
    state.on_frame(
        {"method": "Network.loadingFinished", "params": {"requestId": rid}}, now
    )


def _loaded(state, now):
    state.on_frame({"method": "Page.loadEventFired", "params": {}}, now)


def test_quiet_page_ready_immediately_after_load():
    # Subresources finished well before load: no mandatory post-load wait.
    s = _NetIdleState(now=0.0)
    _open(s, "doc", 0.0)
    _close(s, "doc", 0.1)
    _loaded(s, 0.8)
    assert s.ready(0.8), "already-quiet page must be ready the moment load fires"


def test_not_ready_before_load_even_when_quiet():
    s = _NetIdleState(now=0.0)
    assert not s.ready(5.0), "quiet alone is not readiness — load must have fired"


def test_spa_hydration_burst_blocks_then_clears():
    s = _NetIdleState(now=0.0)
    _loaded(s, 0.5)
    for i in range(5):  # hydration burst: 5 concurrent fetches
        _open(s, f"xhr{i}", 0.6)
    assert not s.ready(2.0), ">2 in-flight must hold readiness open"
    for i in range(3):  # down to 2 pending -> quiet clock restarts
        _close(s, f"xhr{i}", 2.0)
    assert not s.ready(2.4), "quiet window must elapse after the burst clears"
    assert s.ready(2.51), "<=2 in-flight for quiet_ms is ready (networkidle2)"


def test_persistent_connections_are_tolerated():
    # The defining networkidle2 case: 2 long-poll/analytics connections that
    # never finish. The old completion-based probe never went quiet here.
    s = _NetIdleState(now=0.0)
    _open(s, "longpoll", 0.0)
    _open(s, "analytics", 0.0)
    _loaded(s, 0.3)
    assert s.ready(0.9), "2 permanently-open connections must not block"


def test_frequent_beacon_completions_do_not_reset_quiet():
    # Beacons completing every 200ms: completions used to reset the timer;
    # in-flight counting stays <=2 throughout, so the quiet clock never stops.
    s = _NetIdleState(now=0.0)
    _loaded(s, 0.0)
    t = 0.0
    for i in range(10):
        _open(s, f"beacon{i}", t)
        _close(s, f"beacon{i}", t + 0.05)
        t += 0.2
    assert s.ready(0.51), "sub-500ms beacon traffic must not defeat the quiet window"


class _FakeWs:
    """Minimal CDP websocket: scripted frames with delays, echoes probe replies."""

    def __init__(self, script):
        # script: list of (delay_s, frame_dict) relative to previous frame
        self._queue = asyncio.Queue()
        self._script = script
        self._feeder = None

    async def send(self, raw):
        msg = json.loads(raw)
        if msg.get("id") is not None:  # readyState probe -> not complete yet
            await self._queue.put(
                {"id": msg["id"], "result": {"result": {"value": "loading"}}}
            )
        if self._feeder is None:
            self._feeder = asyncio.ensure_future(self._feed())

    async def _feed(self):
        for delay, frame in self._script:
            await asyncio.sleep(delay)
            await self._queue.put(frame)

    async def recv(self):
        return json.dumps(await self._queue.get())


def _run(coro):
    return asyncio.run(coro)


def test_watcher_exits_promptly_despite_beacons():
    # Load fires at 100ms, then beacons complete every 100ms forever (well,
    # 30 of them). Old behavior: wait the full cap. New: ready ~600ms.
    script = [(0.1, {"method": "Page.loadEventFired", "params": {}})]
    for i in range(30):
        script.append(
            (
                0.05,
                {
                    "method": "Network.requestWillBeSent",
                    "params": {"requestId": f"b{i}"},
                },
            )
        )
        script.append(
            (
                0.05,
                {
                    "method": "Network.loadingFinished",
                    "params": {"requestId": f"b{i}"},
                },
            )
        )

    async def scenario():
        ws = _FakeWs(script)
        t0 = time.monotonic()
        await _wait_load_and_network_idle(ws, [0], cap_ms=12_000)
        return time.monotonic() - t0

    elapsed = _run(scenario())
    assert elapsed < 3.0, f"beacon page must not ride the 12s cap (took {elapsed:.2f}s)"


def test_watcher_respects_hard_cap_when_never_quiet():
    # 5 requests open immediately and never finish: must exit at the cap,
    # not hang.
    script = [(0.0, {"method": "Page.loadEventFired", "params": {}})] + [
        (
            0.0,
            {"method": "Network.requestWillBeSent", "params": {"requestId": f"r{i}"}},
        )
        for i in range(5)
    ]

    async def scenario():
        ws = _FakeWs(script)
        t0 = time.monotonic()
        await _wait_load_and_network_idle(ws, [0], cap_ms=1_000)
        return time.monotonic() - t0

    elapsed = _run(scenario())
    assert 0.9 <= elapsed < 2.0, f"expected ~1s cap exit, took {elapsed:.2f}s"


def test_watcher_handles_preloaded_page_via_readystate_probe():
    # load fired before the watcher attached (instant file:// render): the
    # readyState probe must catch it; no Page.loadEventFired ever arrives.
    class _CompleteWs(_FakeWs):
        async def send(self, raw):
            msg = json.loads(raw)
            if msg.get("id") is not None:
                await self._queue.put(
                    {"id": msg["id"], "result": {"result": {"value": "complete"}}}
                )

    async def scenario():
        ws = _CompleteWs([])
        t0 = time.monotonic()
        await _wait_load_and_network_idle(ws, [0], cap_ms=5_000)
        return time.monotonic() - t0

    elapsed = _run(scenario())
    assert elapsed < 1.5, (
        f"preloaded page must exit after one quiet window ({elapsed:.2f}s)"
    )
