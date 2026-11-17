import asyncio
import json
from types import SimpleNamespace

import pytest

from vrbench.llm import LLM, BudgetExceeded, ModelRefused, price


class FakeMessages:
    def __init__(self, text="hello", stop_reason="end_turn", tokens=(1000, 200)) -> None:
        self.text, self.stop_reason, self.tokens = text, stop_reason, tokens
        self.requests: list[dict] = []

    async def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.text),
            ],
            usage=SimpleNamespace(input_tokens=self.tokens[0], output_tokens=self.tokens[1]),
            stop_reason=self.stop_reason,
        )


def make_llm(tmp_path, messages=None, **kwargs):
    messages = messages or FakeMessages()
    llm = LLM(
        client=SimpleNamespace(messages=messages),
        cache_dir=tmp_path / "cache",
        ledger_path=tmp_path / "spend.json",
        **kwargs,
    )
    return llm, messages


def ask(llm, prompt="hi", **kwargs):
    return llm.complete(model="claude-sonnet-5-5", system="sys", prompt=prompt, **kwargs)


def test_price_uses_per_million_rates():
    assert price("claude-sonnet-5-5", 1_000_000, 1_000_000) == 12.0
    assert price("claude-haiku-5-5", 10_000, 1_000) == pytest.approx(0.0015)
    with pytest.raises(ValueError, match="No price configured"):
        price("some-other-model", 1, 1)


async def test_call_is_priced_and_recorded_in_the_ledger(tmp_path):
    llm, messages = make_llm(tmp_path)

    result = await ask(llm)

    assert result.text == "hello"  # thinking blocks are skipped
    assert result.cost_usd == pytest.approx(1000 * 2 / 1e6 + 200 * 10 / 1e6)
    saved = json.loads((tmp_path / "spend.json").read_text())
    assert saved["calls"] == 1 and saved["total_usd"] == pytest.approx(0.004)
    assert saved["by_model"]["claude-sonnet-5-5"]["input_tokens"] == 1000
    assert messages.requests[0]["output_config"] == {"effort": "low"}


async def test_schema_is_sent_as_a_json_output_format(tmp_path):
    llm, messages = make_llm(tmp_path, FakeMessages(text='{"ok": true}'))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    result = await ask(llm, schema=schema)

    assert result.json() == {"ok": True}
    assert messages.requests[0]["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


async def test_repeat_request_is_served_from_cache_for_free(tmp_path):
    llm, messages = make_llm(tmp_path)

    first = await ask(llm)
    second = await ask(llm)
    other = await ask(llm, prompt="something else")

    assert (first.cached, second.cached, other.cached) == (False, True, False)
    assert second.text == first.text and second.latency_s == first.latency_s
    assert len(messages.requests) == 2
    assert llm.ledger.data["calls"] == 2


async def test_budget_stops_calls_and_holds_across_instances(tmp_path):
    llm, messages = make_llm(tmp_path, budget_usd=0.005)

    await ask(llm, prompt="one")  # $0.004
    await ask(llm, prompt="two")  # $0.008, the call that crosses the cap still completes
    with pytest.raises(BudgetExceeded, match=r"\$0.01 of the \$0.01 budget"):
        await ask(llm, prompt="three")
    assert len(messages.requests) == 2

    reopened, messages = make_llm(tmp_path, budget_usd=0.005)  # same ledger file
    with pytest.raises(BudgetExceeded):
        await ask(reopened, prompt="four")
    assert messages.requests == []
    assert (await ask(reopened, prompt="one")).cached  # cached answers stay available


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
async def test_unusable_reply_raises_but_is_still_charged(tmp_path, stop_reason):
    llm, _ = make_llm(tmp_path, FakeMessages(stop_reason=stop_reason))

    with pytest.raises(ModelRefused, match=stop_reason):
        await ask(llm)

    assert llm.ledger.data["calls"] == 1
    assert not list((tmp_path / "cache").glob("*.json"))  # failures are not cached


class StallingMessages(FakeMessages):
    """Hangs on the first ``stalls`` requests, the way a dead connection does."""

    def __init__(self, stalls: int) -> None:
        super().__init__()
        self.stalls = stalls

    async def create(self, **request):
        if len(self.requests) < self.stalls:
            self.requests.append(request)
            await asyncio.sleep(60)
        return await super().create(**request)


async def test_stalled_request_times_out_and_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_backoff(asyncio.sleep))
    llm, messages = make_llm(tmp_path, StallingMessages(stalls=2), timeout_s=0.05)

    result = await ask(llm)

    assert result.text == "hello" and len(messages.requests) == 3
    assert result.latency_s < 0.05  # the successful attempt only, not the two that stalled
    assert llm.ledger.data["calls"] == 1


async def test_gives_up_after_the_last_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_backoff(asyncio.sleep))
    llm, messages = make_llm(tmp_path, StallingMessages(stalls=9), timeout_s=0.02, retries=2)

    with pytest.raises(TimeoutError):
        await ask(llm)

    assert len(messages.requests) == 3 and llm.ledger.data["calls"] == 0


def _no_backoff(real_sleep):
    """Skip the backoff waits between retries, but let a stalled request keep sleeping."""

    async def sleep(seconds):
        await real_sleep(seconds if seconds >= 60 else 0)

    return sleep
