"""Offline SDK transport coverage; run with the frozen eval dependencies."""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")

_LIB = Path(__file__).parents[1] / "eval" / "lib"
if "lib" not in sys.modules:
    _pkg = types.ModuleType("lib")
    _pkg.__path__ = [str(_LIB)]
    sys.modules["lib"] = _pkg

from lib.llm import LLMClient, build_messages, build_react_messages
from lib.model_config import get_model_config
from lib.retrieval import RetrievalResult


@pytest.fixture
def captured_requests(monkeypatch):
    requests = []
    sdk_client = openai.AsyncOpenAI

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": json.loads(request.content)["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "<search>next evidence</search>",
                            "reasoning_content": "Keep this reasoning for the next turn.",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    def client(**kwargs):
        return sdk_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-test-key")
    monkeypatch.setenv("API_KEY", "offline-fallback-key")
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_ANTHROPIC_API_BASE", raising=False)
    return requests


def reader(model, **kwargs):
    config = get_model_config(model)
    return LLMClient(
        **{key: config[key] for key in ("model", "api_base", "api_key")},
        **kwargs,
    )


async def generate_and_close(client, messages):
    try:
        return await client.generate(messages)
    finally:
        await client.client.close()


@pytest.mark.parametrize(
    "region,host",
    [
        ("global_en", "api.minimax.io"),
        ("cn_zh", "api.minimaxi.com"),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        "MiniMax-M3.1-Flash",
        "MiniMax/MiniMax-M3.1-Flash",
        "MiniMax-M3",
        "MiniMax-M2.7",
    ],
)
def test_minimax_serializes_regional_requests(
    monkeypatch, captured_requests, region, host, model
):
    monkeypatch.setenv("MINIMAX_API_REGION", region)
    messages = build_messages(
        "Read the evidence", RetrievalResult(base64_image="aW1hZ2U=")
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "<think>Existing reasoning</think>",
                "reasoning_content": "Existing separate reasoning",
                "tool_calls": [
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"evidence"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_search",
                "content": "Evidence found",
            },
            {"role": "user", "content": "Answer now"},
        ]
    )
    text, usage = asyncio.run(generate_and_close(reader(model), messages))
    request = captured_requests[0]
    payload = json.loads(request.content)
    assert request.url == f"https://{host}/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer offline-test-key"
    assert payload["model"] == model.rsplit("/", 1)[-1]
    assert payload["messages"] == messages
    assert payload["max_tokens"] == 16384
    assert (
        not {
            "thinking",
            "reasoning_effort",
            "service_tier",
            "reasoning_split",
            "chat_template_kwargs",
        }
        & payload.keys()
    )
    assert text == "<search>next evidence</search>"
    assert usage["total_tokens"] == 15


@pytest.mark.parametrize(
    "model,thinking,expected",
    [
        ("MiniMax-M3.1-Flash", True, "adaptive"),
        ("MiniMax-M3", True, "adaptive"),
        ("MiniMax-M3", False, "disabled"),
    ],
)
def test_minimax_explicit_thinking(captured_requests, model, thinking, expected):
    asyncio.run(
        generate_and_close(
            reader(model, enable_thinking=thinking),
            [{"role": "user", "content": "Hello"}],
        )
    )
    payload = json.loads(captured_requests[0].content)
    assert payload["thinking"] == {"type": expected}
    assert "chat_template_kwargs" not in payload


def test_flash_rejects_disabled_thinking_before_request(captured_requests):
    with pytest.raises(ValueError, match="requires adaptive thinking"):
        reader("MiniMax-M3.1-Flash", enable_thinking=False)
    assert captured_requests == []


@pytest.mark.parametrize(
    "model,base,env_name",
    [
        ("MiniMax-M3.1-Flash", "https://gateway.example/custom/v1", "MINIMAX_API_BASE"),
        ("MiniMax-M2.7-highspeed", "http://localhost:9000/v1", "API_BASE"),
        ("custom-reader", "http://localhost:9000/v1", "API_BASE"),
    ],
)
def test_explicit_custom_routes_keep_existing_parameters(
    monkeypatch, captured_requests, model, base, env_name
):
    monkeypatch.setenv(env_name, base)
    asyncio.run(
        generate_and_close(
            reader(model, enable_thinking=False), [{"role": "user", "content": "Hello"}]
        )
    )
    assert captured_requests[0].url == f"{base}/chat/completions"
    payload = json.loads(captured_requests[0].content)
    assert payload["model"] == model
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking" not in payload


def test_flash_react_replays_complete_assistant_message(captured_requests):
    async def run():
        client = reader("MiniMax-M3.1-Flash")
        evidence = RetrievalResult(images=[("a" * 300, 1.0)])
        try:
            text, usage = await client.generate(
                build_react_messages(
                    "Question", [evidence], [], encode_image_fn=lambda _: "unused"
                )
            )
            assert text == "<search>next evidence</search>"
            # Mirrors run_bench: read without mutating, and the recorded
            # token dict must not carry the conversation message.
            history = usage.get("_assistant_message", text)
            assert [k for k in usage if k.startswith("_")] == ["_assistant_message"]
            assert {k for k in usage if not k.startswith("_")} <= {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            }
            await client.generate(
                build_react_messages(
                    "Question",
                    [evidence, evidence],
                    [history],
                    encode_image_fn=lambda _: "unused",
                )
            )
        finally:
            await client.client.close()

    asyncio.run(run())
    payload = json.loads(captured_requests[1].content)
    assert payload["messages"][2] == {
        "role": "assistant",
        "content": "<search>next evidence</search>",
        "reasoning_content": "Keep this reasoning for the next turn.",
    }
    assert payload["messages"][1]["content"][1]["type"] == "image_url"
    assert payload["messages"][3]["content"][1]["type"] == "image_url"
