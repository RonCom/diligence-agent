"""Tool interface, argument validation, caching and timeouts.

Every tool declares a pydantic Args model. The model is the single source for the JSON schema the chat model sees
(Ollama native tools or LangChain), and for validating what the model sends back. A validation failure comes back as
an error message the agent can show the model for one repair try.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from diligence_agent.cache import ToolCache, cache_key, normalize
from diligence_agent.evidence import Evidence
from diligence_agent.http import HttpClient, HttpError

# Calls that time out keep running in their worker thread until the HTTP timeout ends them; the agent moves on.
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


class ToolConfigError(RuntimeError):
    """A tool can't run as configured, e.g. SEC_USER_AGENT is missing."""


@dataclass
class ToolContext:
    http: HttpClient
    data_dir: Path
    fake: bool = False
    env: dict[str, str] = field(default_factory=dict)
    memo: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(
        default_factory=dict
    )  # config.yaml  # per-process data such as the parsed Orange Book


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    version: ClassVar[int] = 1  # bump when the records a tool returns change shape or content
    cacheable: ClassVar[bool] = True  # False for local tools whose data changes when rebuilt

    def run(self, args: BaseModel, ctx: ToolContext) -> list[Evidence]:
        raise NotImplementedError

    def parameters(self) -> dict:
        return _strip_titles(self.Args.model_json_schema())

    def ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters()},
        }


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


@dataclass
class ToolOutcome:
    tool: str
    args: dict[str, Any]  # validated and dumped, or the raw arguments when validation failed
    ok: bool
    evidence: list[Evidence] = field(default_factory=list)
    error: str | None = None
    cache_hit: bool = False
    latency_ms: float = 0.0
    key: str | None = None  # cache key; also the identity loop detection compares

    def as_message(self, max_chars: int = 12000) -> str:
        """What the chat model reads back as the tool result."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if not self.evidence:
            return "No records found."
        lines = [f"[{e.key}] {e.text} ({e.url})" for e in self.evidence]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid arguments: " + "; ".join(parts)


class ToolRunner:
    def __init__(self, tools: list[Tool], ctx: ToolContext, cache: ToolCache | None = None, timeout_s: float = 60):
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx
        self.cache = cache
        self.timeout_s = timeout_s

    def schemas(self) -> list[dict]:
        return [t.ollama_schema() for t in self.tools.values()]

    def call(self, name: str, raw_args: dict | str | None) -> ToolOutcome:
        start = time.perf_counter()

        def done(**kw) -> ToolOutcome:
            return ToolOutcome(tool=name, latency_ms=(time.perf_counter() - start) * 1000, **kw)

        tool = self.tools.get(name)
        if tool is None:
            return done(args={}, ok=False, error=f"unknown tool {name!r}; available: {', '.join(self.tools)}")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as exc:
                return done(args={"_raw": raw_args}, ok=False, error=f"arguments are not valid JSON: {exc}")
        raw_args = raw_args or {}
        try:
            args = tool.Args.model_validate(raw_args)
        except ValidationError as exc:
            return done(args=raw_args, ok=False, error=format_validation_error(exc))

        dumped = args.model_dump(mode="json")
        key = cache_key(name, dumped, tool.version)
        cache = self.cache if tool.cacheable else None
        if cache is not None and (hit := cache.get(key)) is not None:
            evidence = [Evidence.model_validate(r) for r in hit]
            return done(args=normalize(dumped), ok=True, evidence=evidence, cache_hit=True, key=key)

        future = _EXECUTOR.submit(tool.run, args, self.ctx)
        try:
            evidence = future.result(timeout=self.timeout_s)
        except FutureTimeout:
            return done(args=normalize(dumped), ok=False, error=f"timed out after {self.timeout_s:.0f}s", key=key)
        except (HttpError, ToolConfigError) as exc:
            return done(args=normalize(dumped), ok=False, error=str(exc), key=key)

        if cache is not None:
            cache.put(key, [e.model_dump(mode="json") for e in evidence])
        return done(args=normalize(dumped), ok=True, evidence=evidence, key=key)
