import os
import pytest

from app.gemini import call_gemini


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
async def test_call_gemini_real():
    msgs = [{"role": "user", "content": "Diga olá em português, de forma curta."}]
    resp = await call_gemini(msgs, config=None)
    assert isinstance(resp, dict)
    assert "choices" in resp
    assert isinstance(resp["choices"][0]["message"]["content"], str)
