import asyncio
import types
import pytest

from app.gemini import AsyncGeminiClient, call_gemini


class _FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("http error")

    def json(self):
        return self._data


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_call_gemini_normalizes_and_returns(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async def fake_post(url, json=None):
        return _FakeResponse({"candidates": [{"output": "Resposta de teste"}]})

    client = AsyncGeminiClient()
    # patch the instance http client post
    client._client.post = fake_post

    # call internal _create to get NonStreamResponse
    resp = run_async(client._create(messages=[{"role": "user", "content": "Oi"}], stream=False))
    assert hasattr(resp, "choices")
    assert resp.choices[0].message.content == "Resposta de teste"


def test_truncation_and_keep_last(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured = {}

    async def fake_post(url, json=None):
        # capture prompt text
        captured["prompt"] = json.get("prompt", {}).get("text", "")
        return _FakeResponse({"candidates": [{"output": "ok"}]})

    monkeypatch.setattr(AsyncGeminiClient, "__init__", lambda self, base_url=None: None)
    client = AsyncGeminiClient()
    # set required attributes manually because __init__ was bypassed
    client.api_key = "test-key"
    client.base_url = "https://example"
    client._client = types.SimpleNamespace()
    client._client.post = fake_post
    # provide a minimal tokenizer with encode method
    client._tokenizer = types.SimpleNamespace(encode=lambda s: [0] * max(1, len(s)))

    # build 12 messages
    msgs = []
    for i in range(1, 13):
        msgs.append({"role": "user", "content": f"m{i}"})

    # call _create which will enforce KEEP_LAST=6
    run_async(client._create(messages=msgs, stream=False))

    prompt = captured.get("prompt", "")
    # last 6 messages are m7..m12
    assert "m7" in prompt and "m12" in prompt
    # ensure exact '[user] m1' token not present (avoid false match with m10/m11)
    import re
    assert not re.search(r"\[user\] m1($|\s)", prompt)
