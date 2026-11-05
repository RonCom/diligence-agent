"""Chat client for the custom agent: Ollama's /api/chat with native tool calling and JSON-schema outputs.

Each reply carries its token counts and Ollama's timings, so every agent step can be traced and budgeted. The same
interface takes a scripted fake for tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"name": str, "arguments": dict | str}]
    input_tokens: int = 0
    output_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)  # ollama.load_ms, prompt_eval_ms, eval_ms

    def json(self) -> dict:
        text = self.content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            return json.loads(text[start : end + 1]) if start >= 0 and end > start else {}


class OllamaChat:
    def __init__(self, url: str, model: str, temperature: float = 0, num_ctx: int = 16384, timeout_s: float = 600):
        self.url, self.model = url.rstrip("/"), model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.client = httpx.Client(timeout=timeout_s)

    def __call__(self, messages: list[dict], tools: list[dict] | None = None, schema: dict | None = None) -> Reply:
        body: dict[str, Any] = {"model": self.model, "stream": False, "think": False, "options": self.options,
                                "messages": messages}  # fmt: skip
        if tools:
            body["tools"] = tools
        if schema:
            body["format"] = schema
        r = self.client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        j = r.json()
        msg = j.get("message", {})
        calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
                 for c in msg.get("tool_calls") or []]  # fmt: skip
        ns = {"load_duration": "load_ms", "prompt_eval_duration": "prompt_eval_ms", "eval_duration": "eval_ms"}
        return Reply(
            content=msg.get("content", ""),
            tool_calls=calls,
            input_tokens=int(j.get("prompt_eval_count") or 0),
            output_tokens=int(j.get("eval_count") or 0),
            timings={f"ollama.{v}": float(j[k]) / 1e6 for k, v in ns.items() if j.get(k)},
        )


class ScriptedChat:
    """Replays replies in order; a callable entry gets the messages and returns a Reply. Records every request."""

    def __init__(self, replies: list[Reply | Callable[[list[dict]], Reply]]):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.model = "scripted"

    def __call__(self, messages, tools=None, schema=None) -> Reply:
        self.requests.append({"messages": messages, "tools": tools, "schema": schema})
        if not self.replies:
            return Reply(content="")
        nxt = self.replies.pop(0)
        return nxt(messages) if callable(nxt) else nxt
