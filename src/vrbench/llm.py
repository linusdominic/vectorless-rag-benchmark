"""Model calls with three things a benchmark needs: a spending cap, a cache, and metering.

Every call is priced from the token usage the API reports and added to a ledger on disk.
A call is refused once the ledger reaches the budget. Responses are cached by request, so
re-running a benchmark costs nothing and returns the same answers.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Failures worth retrying: a stalled or dropped connection, a rate limit, a server error.
TRANSIENT = (
    TimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# USD per million tokens (input, output). Anthropic list prices as of 2026-11-03.
# Haiku's rate is for prompts up to 100K tokens, which every prompt here is.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would go past the spending cap."""


class ModelRefused(RuntimeError):
    """The model declined the request or ran out of output tokens."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool = False

    def json(self) -> dict[str, Any]:
        return json.loads(self.text)


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        input_rate, output_rate = PRICES[model]
    except KeyError:
        raise ValueError(f"No price configured for model {model!r}") from None
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class Ledger:
    """Running total of money spent, kept on disk so the cap holds across runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = (
            json.loads(path.read_text())
            if path.exists()
            else {"total_usd": 0.0, "calls": 0, "by_model": {}}
        )

    @property
    def total(self) -> float:
        return self.data["total_usd"]

    def add(self, model: str, result: LLMResult) -> None:
        entry = self.data["by_model"].setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        )
        entry["calls"] += 1
        entry["input_tokens"] += result.input_tokens
        entry["output_tokens"] += result.output_tokens
        entry["usd"] = round(entry["usd"] + result.cost_usd, 6)
        self.data["calls"] += 1
        self.data["total_usd"] = round(self.data["total_usd"] + result.cost_usd, 6)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


class LLM:
    """Async wrapper around the Anthropic Messages API."""

    def __init__(
        self,
        client: Any = None,
        cache_dir: Path = Path(".cache/llm"),
        ledger_path: Path = Path("results/spend.json"),
        budget_usd: float = 5.0,
        max_concurrency: int = 6,
        timeout_s: float = 90.0,
        retries: int = 4,
    ) -> None:
        if client is None:
            # Retries are handled in _send, so that latency covers one attempt, not the waits.
            client = anthropic.AsyncAnthropic(max_retries=0)
        self._client = client
        self._timeout_s = timeout_s
        self._retries = retries
        self._cache_dir = cache_dir
        self.ledger = Ledger(ledger_path)
        self.budget_usd = budget_usd
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _cache_path(self, request: dict[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return self._cache_dir / f"{digest}.json"

    async def _send(self, request: dict[str, Any]) -> tuple[Any, float]:
        """Send with a hard timeout and retries. Returns the response and its own latency."""
        for attempt in range(self._retries + 1):
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._client.messages.create(**request), timeout=self._timeout_s
                )
                return response, time.perf_counter() - started
            except TRANSIENT as error:
                logger.warning(
                    "%s attempt %d failed after %.0fs with %s (prompt %d chars)",
                    request["model"],
                    attempt + 1,
                    time.perf_counter() - started,
                    type(error).__name__,
                    len(request["system"]) + len(request["messages"][0]["content"]),
                )
                if attempt == self._retries:
                    raise
                await asyncio.sleep(min(2**attempt, 20))
        raise AssertionError("unreachable")

    async def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2000,
        effort: str = "low",
    ) -> LLMResult:
        """One model call. With ``schema`` the reply is JSON that matches it."""
        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }

        cache_path = self._cache_path(request)
        if cache_path.exists():
            return LLMResult(**{**json.loads(cache_path.read_text()), "cached": True})

        async with self._semaphore:
            if self.ledger.total >= self.budget_usd:
                raise BudgetExceeded(
                    f"Spent ${self.ledger.total:.2f} of the ${self.budget_usd:.2f} budget."
                )
            # No server-side model fallback on purpose: a benchmark must know which model
            # answered, so a refusal is recorded as a failure instead of being rerouted.
            response, latency = await self._send(request)

        usage = response.usage
        result = LLMResult(
            text="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=price(model, usage.input_tokens, usage.output_tokens),
            latency_s=latency,
        )
        self.ledger.add(model, result)  # charge for the call even if its output is unusable
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ModelRefused(f"{model} stopped with {response.stop_reason}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({k: v for k, v in asdict(result).items() if k != "cached"})
        )
        return result
