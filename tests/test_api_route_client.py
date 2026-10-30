"""Offline coverage of the optional API Route evaluation reader."""

import ast
import asyncio
import sys
import types
from pathlib import Path
from runpy import run_path
from unittest.mock import AsyncMock

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
    ("error", "failures", "expected_calls"),
    [
        pytest.param(None, 0, 1, id="success"),
        pytest.param(asyncio.TimeoutError(), 1, 2, id="timeout-recovery"),
        pytest.param(
            RuntimeError("Request timed out"), 1, 2, id="sdk-timeout-recovery"
        ),
        pytest.param(RuntimeError("Connection error"), 1, 2, id="connection-recovery"),
        pytest.param(RuntimeError("429"), 1, 2, id="rate-limit-recovery"),
        pytest.param(asyncio.TimeoutError(), 3, 3, id="timeout-limit"),
        pytest.param(RuntimeError("Connection error"), 12, 12, id="connection-limit"),
        pytest.param(RuntimeError("429"), 6, 6, id="rate-limit-limit"),
        pytest.param(RuntimeError("400 bad request"), 1, 1, id="nonretryable-error"),
        pytest.param(
            RuntimeError("500 internal server error"), 1, 1, id="no-sdk-retry"
        ),
    ],
)
def test_api_route_client_uses_shared_retries(
    monkeypatch, error, failures, expected_calls
):
    calls = []
    options = {}
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def create(**kwargs):
        calls.append(kwargs)
        if error is not None and len(calls) <= failures:
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
    if failures < expected_calls:
        text, usage = asyncio.run(client.generate(messages))
        assert text == "4"
        assert usage["total_tokens"] == 4
    else:
        with pytest.raises(type(error)):
            asyncio.run(client.generate(messages))
    assert len(calls) == expected_calls
    assert sleep.await_count == expected_calls - 1
    assert all(call["model"] == "google/gemini-fixture" for call in calls)
    assert all(call["messages"] == messages for call in calls)
    assert options["base_url"] == "https://global.api-route.com/v1"
    assert options["api_key"] == "api-route-test-key"
    assert options["max_retries"] == 0
    assert client.retry_requests
    assert not client.is_gemini


@pytest.mark.parametrize(
    ("api_route", "atlascloud", "expected"),
    [(False, False, True), (True, False, True), (False, True, False)],
)
def test_cli_retry_policy(api_route, atlascloud, expected):
    # Check the actual CLI wiring without importing the dataset/benchmark stack.
    source = (_LIB.parent / "run_bench.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LLMClient"
    ]
    assert len(calls) == 1
    retry = next(kw.value for kw in calls[0].keywords if kw.arg == "retry_requests")
    args = types.SimpleNamespace(api_route=api_route, atlascloud=atlascloud)
    assert (
        eval(compile(ast.Expression(retry), "run_bench.py", "eval"), {"args": args})
        is expected
    )
