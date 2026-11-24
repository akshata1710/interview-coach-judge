"""LLM clients. The engine needs one method: get JSON back that matches a pydantic model."""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class LLMClient(Protocol):
    model: str

    async def structured(self, system: str, user: str, schema: type[BaseModel]) -> dict[str, Any]:
        """Run one completion whose output must match `schema`; return it as a dict."""
        ...


class AnthropicClient:
    """Claude via the Anthropic SDK, using structured outputs and a cached system prompt.

    API version 2026-10-21 (SDK 2.x) rejects temperature/top_p/top_k, so consistency
    comes from the schema, the anchors and double-scoring rather than temperature 0.
    """

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 1024, client: Any = None):
        from anthropic import AsyncAnthropic  # imported lazily so tests don't need the SDK

        self.model = model
        self.max_tokens = max_tokens
        self._client = client or AsyncAnthropic()

    async def structured(self, system: str, user: str, schema: type[BaseModel]) -> dict[str, Any]:
        resp = await self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if resp.stop_reason in ("refusal", "max_tokens") or resp.parsed_output is None:
            raise RuntimeError(f"no usable {schema.__name__} (stop_reason={resp.stop_reason})")
        return resp.parsed_output.model_dump()


class ScriptedClient:
    """Test double: returns queued responses per schema name, and records every request."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]], model: str = "scripted"):
        self.model = model
        self._responses = {k: list(v) for k, v in responses.items()}
        self.requests: list[tuple[str, str]] = []

    async def structured(self, system: str, user: str, schema: type[BaseModel]) -> dict[str, Any]:
        name = schema.__name__
        self.requests.append((name, user))
        key = name
        if name == "Judgment":
            # route by dimension so parallel calls stay deterministic
            key = f"Judgment:{user.split('(id: ', 1)[1].split(')', 1)[0]}"
        queue = self._responses.get(key)
        if not queue:
            raise RuntimeError(f"no scripted response left for {key}")
        return queue.pop(0)
