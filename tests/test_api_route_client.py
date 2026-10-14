"""Offline coverage of the optional API Route evaluation reader."""

import asyncio
import sys
import types
from pathlib import Path
from runpy import run_path

import pytest

_LIB = Path(__file__).parents[1] / "eval" / "lib"
if "lib" not in sys.modules:
    _pkg = types.ModuleType("lib")
    _pkg.__path__ = [str(_LIB)]
    sys.modules["lib"] = _pkg

from lib.llm import LLMClient

get_config = run_path(str(_LIB / "model_config.py"))["get_api_route_config"]


@pytest.mark.parametrize("model", ["gpt-5.5", "google/gemini-fixture"])
def test_api_route_preserves_model_and_isolates_credentials(monkeypatch, model):
    monkeypatch.setenv("API_ROUTE_API_KEY", "api-route-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-key")
    monkeypatch.setenv("API_KEY", "generic-key")
    config = get_config(model)
    assert config == {
        "model": model,
        "api_base": "https://global.api-route.com/v1",
        "api_key": "api-route-test-key",
    }
    assert get_config(model, "explicit-key")["api_key"] == "explicit-key"


@pytest.mark.parametrize("key", [None, "", " ", "dummy"])
def test_api_route_requires_own_key(monkeypatch, key):
    monkeypatch.delenv("API_ROUTE_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("API_ROUTE_API_KEY", key)
    monkeypatch.setenv("API_KEY", "generic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(ValueError, match="API_ROUTE_API_KEY"):
        get_config("deepseek-ai/deepseek-v3.2")


@pytest.mark.parametrize(
    "error",
    [
        None,
        asyncio.TimeoutError(),
        RuntimeError("Connection error"),
        RuntimeError("429"),
    ],
)
def test_api_route_client_routes_once(monkeypatch, error):
    calls = []
    options = {}

    async def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="4"))],
            usage=types.SimpleNamespace(
                prompt_tokens=3, completion_tokens=1, total_tokens=4
            ),
        )

    def openai_client(**kwargs):
        options.update(kwargs)
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=openai_client)
    )
    client = LLMClient(
        **get_config("google/gemini-fixture", "api-route-test-key"),
        force_openai_compat=True,
        retry_requests=False,
        max_tokens=64,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "2+2?"},
            ],
        }
    ]
    if error is None:
        text, usage = asyncio.run(client.generate(messages))
        assert text == "4"
        assert usage["total_tokens"] == 4
    else:
        with pytest.raises(type(error)):
            asyncio.run(client.generate(messages))
    assert len(calls) == 1
    assert calls[0]["model"] == "google/gemini-fixture"
    assert calls[0]["messages"] == messages
    assert options["base_url"] == "https://global.api-route.com/v1"
    assert options["api_key"] == "api-route-test-key"
    assert options["max_retries"] == 0
    assert not client.is_gemini
