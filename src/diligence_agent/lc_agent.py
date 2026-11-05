"""A LangChain tool-calling agent over the same tools, for comparison with the custom multi-agent loop.

LangChain's create_agent runs a ReAct-style loop: the model calls tools until it writes a final answer. Each tool is
a StructuredTool whose schema comes from the tool's pydantic Args and whose body is ToolRunner.call, so caching,
timeouts, retries and argument validation are shared with the custom agent. A validation error goes back to the
model as the tool result, which is its one repair chance. ModelCallLimitMiddleware enforces the step budget, and a
repeated call with the same normalized arguments returns an error instead of running again.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from diligence_agent.evidence import Evidence
from diligence_agent.tools.base import ToolOutcome, ToolRunner

DECLINE = "NOT IN SOURCES"
LIMIT_PREFIX = "Model call limits exceeded"
CITATION = re.compile(r"\[([a-z_]+:[^\]\s]+)\]")

SYSTEM_PROMPT = """You answer drug diligence questions for an investment team. Today is {today}.
Use the tools to find records. Answer only from tool results.
After every claim, cite the record it comes from in square brackets, exactly as the tool printed its key,
for example [orange_book:N203415/patent/7709517].
If two sources disagree, give both values with their citations.
If the records don't answer the question, reply exactly: {decline}
Keep the answer to a few sentences."""


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

    @property
    def evidence(self) -> dict[str, Evidence]:
        return {e.key: e for c in self.calls for e in c.evidence}

    @property
    def cited(self) -> list[str]:
        return CITATION.findall(self.answer)

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
            "latency_s": round(self.latency_s, 1),
        }


def as_langchain_tools(runner: ToolRunner, result: LcResult) -> list[StructuredTool]:
    seen: set[str] = set()

    def make(name: str):
        def call(**kwargs: Any) -> str:
            outcome = runner.call(name, kwargs)
            if outcome.key is not None and outcome.key in seen:
                result.repeated_calls += 1
                return "ERROR: you already made this call with these arguments. Use the records you have."
            if outcome.key is not None:
                seen.add(outcome.key)
            result.calls.append(outcome)
            return outcome.as_message()

        return call

    return [
        StructuredTool.from_function(
            func=make(t.name), name=t.name, description=t.description, args_schema=t.parameters()
        )
        for t in runner.tools.values()
    ]


def build_model(cfg: dict) -> ChatOllama:
    return ChatOllama(
        model=cfg.get("chat_model", "gemma4:e4b"),
        base_url=cfg.get("ollama_url", "http://localhost:11434"),
        temperature=cfg.get("temperature", 0),
        reasoning=False,
        num_ctx=cfg.get("num_ctx", 16384),
    )


def run_question(question: str, runner: ToolRunner, cfg: dict, model: Any = None) -> LcResult:
    result = LcResult(question=question, answer="")
    agent = create_agent(
        model=model or build_model(cfg),
        tools=as_langchain_tools(runner, result),
        system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat(), decline=DECLINE),
        middleware=[
            ModelCallLimitMiddleware(run_limit=cfg.get("budgets", {}).get("max_steps", 12), exit_behavior="end")
        ],
    )
    start = time.perf_counter()
    state = agent.invoke({"messages": [{"role": "user", "content": question}]})
    result.latency_s = time.perf_counter() - start
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
    return result


def _text(message: AIMessage) -> str:
    return message.content if isinstance(message.content, str) else message.text
