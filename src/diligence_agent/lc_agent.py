"""A LangChain tool-calling agent over the same tools, for comparison with the custom multi-agent loop.

LangChain's create_agent runs a ReAct-style loop: the model calls tools until it writes a final answer. Each tool is
a StructuredTool whose schema comes from the tool's pydantic Args and whose body is ToolRunner.call, so caching,
timeouts, retries and argument validation are shared with the custom agent. A validation error goes back to the
model as the tool result, which is its one repair chance. ModelCallLimitMiddleware enforces the step budget, and a
repeated call with the same normalized arguments returns an error instead of running again.

TracingMiddleware puts one MLflow span on each model call and each tool call, inside one root span per question,
with the same attributes the custom agent's spans carry (tokens, cost, latency, model, prompt hash, cache hit).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import mlflow
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from diligence_agent.evidence import Evidence, citations
from diligence_agent.tools.base import ToolOutcome, ToolRunner
from diligence_agent.tracing import SpanType, cost, model_span_attributes, ollama_timings, sha

DECLINE = "NOT IN SOURCES"
LIMIT_PREFIX = "Model call limits exceeded"

SYSTEM_PROMPT = """You answer drug diligence questions for an investment team. Today is {today}.
Use the tools to find records. Answer only from tool results.
After every claim, cite the record it comes from in square brackets, exactly as the tool printed its key,
for example [orange_book:N203415/patent/7709517].
If two sources disagree, give both values with their citations.
If the records don't answer the question, reply exactly: {decline}
Keep the answer to a few sentences."""
PROMPT_SHA = sha(SYSTEM_PROMPT)


@dataclass
class LcResult:
    question: str
    answer: str
    calls: list[ToolOutcome] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    repeated_calls: int = 0
    budget_stop: bool = False
    cost_usd: float = 0.0
    trace_id: str | None = None
    by_call: dict[tuple[str, str], ToolOutcome | None] = field(default_factory=dict)  # None marks a refused repeat

    @property
    def evidence(self) -> dict[str, Evidence]:
        return {e.key: e for c in self.calls for e in c.evidence}

    @property
    def cited(self) -> list[str]:
        return citations(self.answer)

    @property
    def declined(self) -> bool:
        return self.answer.strip().upper().startswith(DECLINE)

    def summary(self) -> dict[str, Any]:
        cited = self.cited
        known = self.evidence
        return {
            "declined": self.declined,
            "model_calls": self.model_calls,
            "tool_calls": [
                {
                    "tool": c.tool,
                    "args": c.args,
                    "ok": c.ok,
                    "error": c.error,
                    "records": len(c.evidence),
                    "cache_hit": c.cache_hit,
                    "latency_ms": round(c.latency_ms),
                }
                for c in self.calls
            ],
            "tool_call_validity": (sum(c.ok for c in self.calls) / len(self.calls)) if self.calls else None,
            "repeated_calls": self.repeated_calls,
            "budget_stop": self.budget_stop,
            "citations": len(cited),
            "citations_to_unseen_records": [k for k in cited if k not in known],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": round(self.latency_s, 1),
            "trace_id": self.trace_id,
        }


def as_langchain_tools(runner: ToolRunner, result: LcResult) -> list[StructuredTool]:
    seen: set[str] = set()

    def make(name: str):
        def call(**kwargs: Any) -> str:
            outcome = runner.call(name, kwargs)
            if outcome.key is not None and outcome.key in seen:
                result.repeated_calls += 1
                result.by_call[(name, call_key(kwargs))] = None
                return "ERROR: you already made this call with these arguments. Use the records you have."
            if outcome.key is not None:
                seen.add(outcome.key)
            result.calls.append(outcome)
            result.by_call[(name, call_key(kwargs))] = outcome
            return outcome.as_message()

        return call

    return [
        StructuredTool.from_function(
            func=make(t.name), name=t.name, description=t.description, args_schema=t.parameters()
        )
        for t in runner.tools.values()
    ]


def call_key(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)


class TracingMiddleware(AgentMiddleware):
    """One MLflow span per model call (an agent step) and per tool call."""

    def __init__(self, result: LcResult, model: str, pricing: dict | None) -> None:
        super().__init__()
        self.result, self.model, self.pricing = result, model, pricing
        self.step = 0

    def wrap_model_call(self, request, handler):
        self.step += 1
        with mlflow.start_span(name=f"step_{self.step}", span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs({"messages": len(request.messages), "tools": [t.name for t in request.tools or []]})
            t0 = time.perf_counter()
            response = handler(request)
            latency_ms = (time.perf_counter() - t0) * 1000
            if isinstance(response, AIMessage):
                msg = response
            else:
                msg = next((m for m in reversed(getattr(response, "result", [])) if isinstance(m, AIMessage)), None)
            usage = (msg.usage_metadata if msg else None) or {}
            span.set_outputs(
                {
                    "text": _text(msg)[:4000] if msg else "",
                    "tool_calls": [{"name": c["name"], "args": c["args"]} for c in (msg.tool_calls if msg else [])],
                }
            )
            timings = ollama_timings(msg.response_metadata if msg else {})
            span.set_attributes(model_span_attributes(self.model, usage, self.pricing, latency_ms, PROMPT_SHA, timings))
        return response

    def wrap_tool_call(self, request, handler):
        name, args = request.tool_call["name"], request.tool_call.get("args", {})
        with mlflow.start_span(name=f"tool:{name}", span_type=SpanType.TOOL) as span:
            span.set_inputs(args)
            t0 = time.perf_counter()
            response = handler(request)
            outcome = self.result.by_call.get((name, call_key(args)), "missing")
            attrs: dict[str, Any] = {"latency_ms": (time.perf_counter() - t0) * 1000, "tool": name}
            if outcome is None:
                attrs |= {"ok": False, "repeated": True}
            elif isinstance(outcome, ToolOutcome):
                attrs |= {
                    "ok": outcome.ok,
                    "cache_hit": outcome.cache_hit,
                    "records": len(outcome.evidence),
                    "error": outcome.error or "",
                    "tool_latency_ms": outcome.latency_ms,
                    "record_keys": [e.key for e in outcome.evidence[:25]],
                }
            span.set_attributes(attrs)
            content = response.content if isinstance(response, ToolMessage) else str(response)
            span.set_outputs({"content": str(content)[:4000]})
        return response


def build_model(cfg: dict) -> ChatOllama:
    return ChatOllama(
        model=cfg.get("chat_model", "gemma4:e4b"),
        base_url=cfg.get("ollama_url", "http://localhost:11434"),
        temperature=cfg.get("temperature", 0),
        reasoning=False,
        num_ctx=cfg.get("num_ctx", 16384),
    )


def run_question(
    question: str, runner: ToolRunner, cfg: dict, model: Any = None, tags: dict[str, Any] | None = None
) -> LcResult:
    result = LcResult(question=question, answer="")
    model_name = cfg.get("chat_model", "gemma4:e4b")
    max_steps = cfg.get("budgets", {}).get("max_steps", 12)
    agent = create_agent(
        model=model or build_model(cfg),
        tools=as_langchain_tools(runner, result),
        system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat(), decline=DECLINE),
        middleware=[
            TracingMiddleware(result, model_name, cfg.get("pricing")),
            ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end"),
        ],
    )
    with mlflow.start_span(name="lc_agent", span_type=SpanType.AGENT) as root:
        root.set_inputs({"question": question})
        root.set_attributes(
            {"agent": "langchain", "chat_model": model_name, "prompt_sha": PROMPT_SHA, "max_steps": max_steps}
        )
        start = time.perf_counter()
        state = agent.invoke({"messages": [{"role": "user", "content": question}]})
        result.latency_s = time.perf_counter() - start
        _finish(result, state, cfg.get("pricing"), model_name)
        s = result.summary()
        root.set_outputs(
            {
                "answer": result.answer,
                "declined": result.declined,
                "budget_stop": result.budget_stop,
                "citations": result.cited,
            }
        )
        totals = {k: s[k] for k in ("model_calls", "repeated_calls", "input_tokens", "output_tokens", "cost_usd")}
        root.set_attributes(
            totals
            | {
                "latency_ms": result.latency_s * 1000,
                "tool_calls": len(result.calls),
                "citations_to_unseen_records": s["citations_to_unseen_records"],
            }
        )
        trace_id = getattr(root, "trace_id", None)
        if trace_id and str(trace_id).startswith("tr-"):
            result.trace_id = trace_id
            mlflow.update_current_trace(tags={k: str(v) for k, v in (tags or {}).items()} | {"agent": "langchain"})
    return result


def _finish(result: LcResult, state: dict, pricing: dict | None, model_name: str) -> None:
    ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
    # On a budget stop, ModelCallLimitMiddleware appends its own AIMessage; it isn't a model call.
    if ai and _text(ai[-1]).startswith(LIMIT_PREFIX):
        result.budget_stop = True
        ai = ai[:-1]
    result.model_calls = len(ai)
    for m in ai:
        usage = m.usage_metadata or {}
        result.input_tokens += usage.get("input_tokens", 0)
        result.output_tokens += usage.get("output_tokens", 0)
    final = ai[-1] if ai else None
    if result.budget_stop or final is None or final.tool_calls:
        result.budget_stop = True
        result.answer = f"{DECLINE} (step budget reached)"
    else:
        result.answer = _text(final)
    result.cost_usd = cost(pricing, model_name, result.input_tokens, result.output_tokens)["total_cost"]


def _text(message: AIMessage) -> str:
    return message.content if isinstance(message.content, str) else message.text
