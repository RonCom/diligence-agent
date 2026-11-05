"""The custom multi-agent loop: planner, researcher, writer, verifier.

  planner     one structured call: 1-4 research steps, each with a goal and a tool. Its out-of-scope flag is
              logged only: in a smoke run it declined an answerable 10-K question, so the writer decides
  researcher  native tool calling, per plan step, through ToolRunner (cache, retries, timeouts, validation);
              an invalid call gets its error back once to repair it
  writer      answers from the collected records only, cites every claim, or declines
  verifier    checks each cited claim: the cited records exist, every date, year and percentage in the claim
              appears in them (code), and the records support it (one structured call). Unsupported claims get
              one more research round and a redraft; what is still unsupported is cut. No claims left: decline.

Budgets per request: model calls, tokens, dollars and wall clock. Research stops early enough to leave calls for the
writer and verifier; a budget stop returns a partial answer. Loop detection: the same tool with the same normalized
arguments twice is refused; three research turns with no new records end research. Every model call and tool call is
an MLflow span with tokens, cost, latency, model and prompt hash.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import mlflow

from diligence_agent.cache import normalize
from diligence_agent.evidence import Evidence, citations, strip_citations
from diligence_agent.grading import DECLINE, dates_in, declined
from diligence_agent.llm import Reply
from diligence_agent.tools.base import ToolOutcome, ToolRunner
from diligence_agent.tracing import SpanType, cost, model_span_attributes, sha

RECORD_CHARS = 1500  # per record shown to the writer and verifier
CONTEXT_CHARS = 24000  # all records shown to the writer
TURNS_PER_STEP = 3
RESERVED_CALLS = 4  # writer, verifier, and one redraft and re-verify

PLANNER = """You plan research for a drug diligence question. Today is {today}. Tools:
{tools}
Rules: ClinicalTrials.gov records use generic drug names, so when the question gives a brand, get the generic
name from drugs_at_fda first. The Orange Book covers small-molecule NDA drugs; biologics (BLAs) have no Orange
Book patents. Royalty Pharma's royalty terms, receipts and end-market sales are in its 10-K (doc_search).
Write 1 to 4 steps. Set out_of_scope to true only when no tool could hold the answer: prices, forecasts, analyst
views, approvals outside the U.S., other companies' financial results, or future events."""
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "out_of_scope": {"type": "boolean"},
        "reason": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "goal": {"type": "string"}, "tool": {"type": "string"}}, "required": ["goal", "tool"]}},
    },
    "required": ["out_of_scope", "steps"],
}  # fmt: skip

RESEARCHER = """You gather records for one research step of a drug diligence question. Today is {today}.
Call tools to find the records the step needs. Use generic drug names for ClinicalTrials.gov. When you have the
records, or the tools can't provide them, stop calling tools and reply with one line saying what you found."""

WRITER = """You answer drug diligence questions for an investment team. Today is {today}.
Answer only from the records below. After every claim, cite the record it comes from in square brackets, exactly
as its key is printed, for example [orange_book:N203415/patent/7709517].
If two records disagree, give both values with their citations.
If the records don't answer the question, reply exactly: {decline}
Keep the answer to a few sentences."""

VERIFIER = """You check whether each claim is supported by the records it cites. A claim is supported only if the
cited records state it; a claim that adds facts, numbers or dates the records don't contain is not supported."""
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "integer"}, "supported": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["claim", "supported"]}}},
    "required": ["results"],
}  # fmt: skip
PROMPT_SHA = sha(PLANNER + RESEARCHER + WRITER + VERIFIER)


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    max_steps: int = 12
    max_tokens: int = 60000
    max_usd: float = 0.0  # 0: not enforced
    max_wall_s: float = 600
    steps: int = 0
    tokens: int = 0
    usd: float = 0.0
    start: float = field(default_factory=time.perf_counter)
    hit: str | None = None

    def charge(self, reply: Reply, usd: float) -> None:
        self.steps += 1
        self.tokens += reply.input_tokens + reply.output_tokens
        self.usd += usd

    def exhausted(self, reserve: int = 0) -> str | None:
        if self.steps + reserve >= self.max_steps:
            return "steps"
        if self.tokens >= self.max_tokens:
            return "tokens"
        if self.max_usd and self.usd >= self.max_usd:
            return "usd"
        if time.perf_counter() - self.start >= self.max_wall_s:
            return "wall_clock"
        return None


@dataclass
class Claim:
    text: str
    cited: list[str]
    code_ok: bool = True
    supported: bool | None = None
    reason: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    plan: list[dict] = field(default_factory=list)
    out_of_scope: bool = False
    evidence: dict[str, Evidence] = field(default_factory=dict)
    calls: list[ToolOutcome] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    claims_cut: int = 0
    research_rounds: int = 1
    repeated_calls: int = 0
    loop_stops: int = 0
    repair_attempts: int = 0
    budget_stop: str | None = None
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    role_ms: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None

    @property
    def cited(self) -> list[str]:
        return citations(self.answer)

    def summary(self) -> dict[str, Any]:
        verified = [c for c in self.claims if c.supported is not None]
        return {
            "declined": bool(declined(self.answer)), "model_calls": self.model_calls,
            "tool_calls": [{"tool": c.tool, "args": c.args, "ok": c.ok, "error": c.error, "records": len(c.evidence),
                            "cache_hit": c.cache_hit, "latency_ms": round(c.latency_ms)} for c in self.calls],
            "repeated_calls": self.repeated_calls, "loop_stops": self.loop_stops,
            "repair_attempts": self.repair_attempts, "budget_stop": self.budget_stop,
            "citations": len(self.cited),
            "citations_to_unseen_records": [k for k in self.cited if k not in self.evidence],
            "claims": len(verified), "claims_supported": sum(bool(c.supported) for c in verified),
            "claims_cut": self.claims_cut, "research_rounds": self.research_rounds,
            "planner_out_of_scope": self.out_of_scope,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_usd": self.cost_usd,
            "latency_s": round(self.latency_s, 2), "role_ms": {k: round(v) for k, v in self.role_ms.items()},
            "trace_id": self.trace_id,
        }  # fmt: skip


# ---- claims ---------------------------------------------------------------------------------------------------------


def split_claims(answer: str) -> list[Claim]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", answer) if s.strip()]
    return [Claim(text=s, cited=citations(s)) for s in sentences]


_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_PCT = re.compile(r"\b\d+(?:\.\d+)?%")


def values_in_records(claim: str, records: list[str]) -> bool:
    """Every date, year and percentage in the claim appears in at least one cited record."""
    text = strip_citations(claim)
    blob = " ".join(records)
    blob_dates = set(dates_in(blob))
    for d in dates_in(text):
        if d not in blob_dates and d not in blob:
            return False
    for y in {m.group(0) for m in _YEAR.finditer(text)}:
        if y not in blob:
            return False
    return all(p in blob for p in {m.group(0) for m in _PCT.finditer(text)})


def code_check_claims(answer: str, records: dict[str, str], given: str = "") -> list[Claim]:
    """The verifier's code check, usable on any system's answer: each claim cites only records that were collected,
    and every date, year and percentage it states appears in them, or in `given` (the question, whose own date
    window an answer may restate). Sets code_ok; leaves `supported` to the LLM."""
    claims = split_claims(answer)
    for c in claims:
        texts = [records[k] for k in c.cited if k in records]
        found = len(texts) == len(c.cited) and values_in_records(c.text, texts + ([given] if given else []))
        c.code_ok = bool(c.cited) and found
        if not c.code_ok:
            c.reason = "uncited" if not c.cited else "cites a record not collected or a value not in it"
    return claims


def strip_unsupported(answer_claims: list[Claim]) -> tuple[str, int]:
    kept = [c.text for c in answer_claims if c.supported]
    return " ".join(kept), sum(not c.supported for c in answer_claims)


# ---- the agent --------------------------------------------------------------------------------------------------


class Agent:
    def __init__(self, runner: ToolRunner, chat, cfg: dict):
        self.runner, self.chat, self.cfg = runner, chat, cfg
        self.model = cfg.get("chat_model", "gemma4:e4b")
        self.pricing = cfg.get("pricing")
        self.today = date.today().isoformat()
        self.tool_list = "\n".join(f"- {t.name}: {t.description}" for t in runner.tools.values())

    # one model call, traced and budgeted
    def _call(self, role: str, r: AgentResult, budget: Budget, messages, tools=None, schema=None) -> Reply:
        with mlflow.start_span(name=role, span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs({"messages": messages[-2:] if len(messages) > 2 else messages})
            t0 = time.perf_counter()
            reply = self.chat(messages, tools=tools, schema=schema)
            ms = (time.perf_counter() - t0) * 1000
            usage = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
            usd = cost(self.pricing, self.model, reply.input_tokens, reply.output_tokens)["total_cost"]
            span.set_outputs({"content": reply.content[:4000], "tool_calls": reply.tool_calls})
            span.set_attributes(model_span_attributes(self.model, usage, self.pricing, ms, PROMPT_SHA,
                                                      reply.timings | {"role": role}))  # fmt: skip
        budget.charge(reply, usd)
        r.model_calls += 1
        r.input_tokens += reply.input_tokens
        r.output_tokens += reply.output_tokens
        r.cost_usd += usd
        base = role.split("_")[0]
        r.role_ms[base] = r.role_ms.get(base, 0.0) + ms
        return reply

    def _tool(self, name: str, args, r: AgentResult, seen: set[str]) -> str:
        with mlflow.start_span(name=f"tool:{name}", span_type=SpanType.TOOL) as span:
            span.set_inputs(args if isinstance(args, dict) else {"raw": args})
            outcome = self.runner.call(name, args)
            attrs: dict[str, Any] = {"tool": name, "ok": outcome.ok, "cache_hit": outcome.cache_hit,
                                     "records": len(outcome.evidence), "error": outcome.error or "",
                                     "latency_ms": outcome.latency_ms}  # fmt: skip
            if outcome.key is not None and outcome.key in seen:
                r.repeated_calls += 1
                span.set_attributes(attrs | {"repeated": True})
                return "ERROR: you already made this call with these arguments. Use the records you have."
            if outcome.key is not None:
                seen.add(outcome.key)
            r.calls.append(outcome)
            r.role_ms["tools"] = r.role_ms.get("tools", 0.0) + outcome.latency_ms
            for e in outcome.evidence:
                r.evidence.setdefault(e.key, e)
            span.set_attributes(attrs | {"record_keys": [e.key for e in outcome.evidence[:25]]})
            message = outcome.as_message(max_chars=6000)
            span.set_outputs({"content": message[:4000]})
            return message

    def plan(self, question: str, r: AgentResult, budget: Budget) -> None:
        msgs = [{"role": "system", "content": PLANNER.format(today=self.today, tools=self.tool_list)},
                {"role": "user", "content": question}]  # fmt: skip
        p = self._call("planner", r, budget, msgs, schema=PLAN_SCHEMA).json()
        r.out_of_scope = bool(p.get("out_of_scope"))
        steps = [s for s in p.get("steps", []) if isinstance(s, dict) and s.get("goal")][:4]
        r.plan = steps or [{"goal": question, "tool": ""}]

    def research(self, question: str, goals: list[dict], r: AgentResult, budget: Budget, seen: set[str]) -> None:
        tools = self.runner.schemas()
        idle = 0
        for step in goals:
            msgs = [{"role": "system", "content": RESEARCHER.format(today=self.today)},
                    {"role": "user", "content": f"Question: {question}\nStep: {step['goal']}"
                     f"{' (suggested tool: ' + step['tool'] + ')' if step.get('tool') else ''}\n"
                     f"Records already collected: {', '.join(list(r.evidence)[:40]) or 'none'}"}]  # fmt: skip
            repaired: set[str] = set()
            for _ in range(TURNS_PER_STEP):
                if hit := budget.exhausted(reserve=RESERVED_CALLS):
                    r.budget_stop = hit
                    return
                before = len(r.evidence)
                reply = self._call(f"researcher_{len(r.calls)}", r, budget, msgs, tools=tools)
                if not reply.tool_calls:
                    break
                msgs.append({"role": "assistant", "content": reply.content,
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in reply.tool_calls]})  # fmt: skip
                for c in reply.tool_calls:
                    result = self._tool(c["name"], c["arguments"], r, seen)
                    if result.startswith("ERROR: invalid arguments") or result.startswith("ERROR: arguments are not"):
                        sig = c["name"]
                        if sig in repaired:  # one repair try per tool per step
                            result += " Do not retry this tool for this step."
                        else:
                            repaired.add(sig)
                            r.repair_attempts += 1
                            result += " Fix the arguments and call the tool again."
                    msgs.append({"role": "tool", "content": result, "tool_name": c["name"]})
                idle = 0 if len(r.evidence) > before else idle + 1
                if idle >= self.cfg.get("loop_detection", {}).get("steps_without_new_evidence", 3):
                    r.loop_stops += 1
                    return

    def records_text(self, r: AgentResult, keys: list[str] | None = None) -> str:
        out, total = [], 0
        for k in keys or list(r.evidence):
            e = r.evidence.get(k)
            if e is None:
                continue
            line = f"[{k}] {e.text[:RECORD_CHARS]}"
            if total + len(line) > CONTEXT_CHARS:
                break
            out.append(line)
            total += len(line)
        return "\n\n".join(out)

    def write(self, question: str, r: AgentResult, budget: Budget, role: str = "writer") -> str:
        if not r.evidence:
            return DECLINE
        # The searches show what each record set was filtered by (a date window, a phase), which the records alone
        # don't say: without them the writer re-derived a 12-month window and got it wrong.
        searches = "\n".join(f"- {c.tool}({json.dumps(c.args, default=str)}) -> {len(c.evidence)} records"
                             for c in r.calls if c.ok)  # fmt: skip
        msgs = [{"role": "system", "content": WRITER.format(today=self.today, decline=DECLINE)},
                {"role": "user", "content": f"Searches run:\n{searches}\n\nRecords:\n\n{self.records_text(r)}\n\n"
                                            f"Question: {question}"}]  # fmt: skip
        return self._call(role, r, budget, msgs).content.strip()

    def verify(self, answer: str, r: AgentResult, budget: Budget, role: str = "verifier") -> list[Claim]:
        claims = code_check_claims(answer, {k: e.text for k, e in r.evidence.items()})
        for c in claims:
            if not c.code_ok:
                c.supported = False
        to_check = [i for i, c in enumerate(claims) if c.code_ok]
        if to_check:
            lines = []
            for n, i in enumerate(to_check, 1):
                cited = "\n".join(f"  [{k}] {r.evidence[k].text[:RECORD_CHARS]}" for k in claims[i].cited)
                lines.append(f"Claim {n}: {strip_citations(claims[i].text)}\nCited records:\n{cited}")
            msgs = [{"role": "system", "content": VERIFIER}, {"role": "user", "content": "\n\n".join(lines)}]
            out = self._call(role, r, budget, msgs, schema=VERIFY_SCHEMA).json().get("results", [])
            verdict = {int(x.get("claim", 0)): x for x in out if isinstance(x, dict)}
            for n, i in enumerate(to_check, 1):
                v = verdict.get(n, {})
                claims[i].supported = bool(v.get("supported", False))
                claims[i].reason = v.get("reason", "no verdict")
        return claims

    def run(self, question: str, tags: dict | None = None) -> AgentResult:
        b = self.cfg.get("budgets", {})
        budget = Budget(max_steps=b.get("max_steps", 12), max_tokens=b.get("max_tokens", 60000),
                        max_usd=b.get("max_usd", 0.0), max_wall_s=b.get("max_wall_clock_s", 600))  # fmt: skip
        r = AgentResult(question=question)
        seen: set[str] = set()
        t0 = time.perf_counter()
        with mlflow.start_span(name="agent", span_type=SpanType.AGENT) as root:
            root.set_inputs({"question": question})
            root.set_attributes({"agent": "custom", "chat_model": self.model, "prompt_sha": PROMPT_SHA,
                                 "max_steps": budget.max_steps})  # fmt: skip
            self._run(question, r, budget, seen)
            r.latency_s = time.perf_counter() - t0
            s = r.summary()
            root.set_outputs({"answer": r.answer, "plan": r.plan, "citations": r.cited})
            root.set_attributes({k: s[k] for k in ("model_calls", "input_tokens", "output_tokens", "cost_usd",
                                                    "claims", "claims_supported", "claims_cut", "repeated_calls",
                                                    "loop_stops", "repair_attempts", "research_rounds")}
                                | {"budget_stop": r.budget_stop or "", "latency_ms": r.latency_s * 1000,
                                   "tool_calls": len(r.calls), "role_ms": json.dumps(s["role_ms"])})  # fmt: skip
            trace_id = getattr(root, "trace_id", None)
            if trace_id and str(trace_id).startswith("tr-"):
                r.trace_id = trace_id
                mlflow.update_current_trace(tags={k: str(v) for k, v in (tags or {}).items()} | {"agent": "custom"})
        return r

    def _run(self, question: str, r: AgentResult, budget: Budget, seen: set[str]) -> None:
        self.plan(question, r, budget)  # out_of_scope is recorded only: a small model's flag ends no request
        self.research(question, r.plan, r, budget, seen)
        draft = self.write(question, r, budget)
        if declined(draft):
            r.answer = DECLINE if declined(draft) == "exact" else draft
            return
        claims = self.verify(draft, r, budget)
        weak = [c for c in claims if not c.supported]
        if weak and not budget.exhausted(reserve=2):
            r.research_rounds = 2
            goals = [{"goal": f"Find records that support or correct this claim: {strip_citations(c.text)}",
                      "tool": ""} for c in weak[:3]]  # fmt: skip
            self.research(question, goals, r, budget, seen)
            draft = self.write(question, r, budget, role="writer_redraft")
            if declined(draft):
                r.answer = DECLINE if declined(draft) == "exact" else draft
                return
            claims = self.verify(draft, r, budget, role="verifier_recheck")
        r.claims = claims
        r.answer, r.claims_cut = strip_unsupported(claims)
        if not r.answer:
            r.answer = DECLINE
        if r.budget_stop is None and (hit := budget.exhausted()):
            r.budget_stop = hit


def build_chat(cfg: dict):
    from diligence_agent.llm import OllamaChat

    return OllamaChat(cfg.get("ollama_url", "http://localhost:11434"), cfg.get("chat_model", "gemma4:e4b"),
                      temperature=cfg.get("temperature", 0), num_ctx=cfg.get("num_ctx", 16384))  # fmt: skip


def run_question(question: str, runner: ToolRunner, cfg: dict, chat=None, tags: dict | None = None) -> AgentResult:
    return Agent(runner, chat or build_chat(cfg), cfg).run(question, tags=tags)


__all__ = ["Agent", "AgentResult", "Budget", "normalize", "run_question", "split_claims", "values_in_records"]
