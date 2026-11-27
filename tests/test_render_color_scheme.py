import asyncio

from pixelrag_render.backends import cdp


def test_capture_uses_the_light_color_scheme_whatever_the_host_theme(monkeypatch):
    sent = []

    async def fake_send(ws, msg_id_ref, method, params=None):
        sent.append((method, params))

    monkeypatch.setattr(cdp, "_cdp_send", fake_send)
    asyncio.run(cdp._setup_page(None, [0], 875, 8192, wait_network_idle=False))

    assert (
        "Emulation.setEmulatedMedia",
        {"features": [{"name": "prefers-color-scheme", "value": "light"}]},
    ) in sent
