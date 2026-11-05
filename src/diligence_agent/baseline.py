"""Single-call RAG baseline: one retrieval, one model call, no tools. The number the agents have to beat.

Strong corpus (pre-registered): the doc_search index (10-K and labels) plus every record the structured tools return
for the 32 products: Orange Book applications, patents and exclusivities, Drugs@FDA approvals, and Phase 3
ClinicalTrials.gov studies found by generic name. Each record is one chunk whose key is the tool's own record key, so
the baseline cites exactly what an agent would and citation metrics compare directly. A win for an agent therefore
means tool use and multi-step research beat retrieval over the same facts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from diligence_agent.corpus import Chunk
from diligence_agent.evidence import Evidence, citations
from diligence_agent.grading import DECLINE
from diligence_agent.retrieval import FakeEmbedder, Index, OllamaEmbedder
from diligence_agent.tools.base import ToolRunner
from diligence_agent.tracing import SpanType, cost, model_span_attributes, ollama_timings, sha

SOURCE_NAMES = {"orange_book": "FDA Orange Book", "drugs_at_fda": "Drugs@FDA", "ctgov": "ClinicalTrials.gov"}
SALTS = {"sodium", "hydrochloride", "bromide", "olamine", "mesylate", "trifenatate", "malate", "s-malate",
         "propanediol", "potassium", "succinate", "citrate", "tartrate", "fumarate", "maleate", "besylate",
         "dihydrochloride", "sulfate", "benzoate"}  # fmt: skip
ACTIVE = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]

SYSTEM_PROMPT = """You answer drug diligence questions for an investment team. Today is {today}.
Answer only from the numbered records below. After every claim, cite the record it comes from in square brackets,
exactly as its key is printed, for example [orange_book:N203415/patent/7709517].
If two records disagree, give both values with their citations.
If the records don't answer the question, reply exactly: {decline}
Keep the answer to a few sentences."""
PROMPT_SHA = sha(SYSTEM_PROMPT)


def generic_query(ingredients: list[str]) -> str | None:
    """First active ingredient without salt words: 'NUSINERSEN SODIUM' -> 'nusinersen'."""
    for raw in ingredients:
        first = re.split(r"[;,]", raw)[0].strip().lower()
        first = re.sub(r"-[a-z]{4}\b", "", first)  # biologic suffixes: tarlatamab-dlle -> tarlatamab
        words = [w for w in first.split() if w not in SALTS]
        if words:
            return " ".join(words)
    return None


def record_chunks(runner: ToolRunner, products: list[dict], log=print) -> list[Chunk]:
    out: dict[str, Chunk] = {}

    def keep(ev: list[Evidence], drug: str) -> int:
        n = 0
        for e in ev:
            if e.key not in out:
                section = e.record_id.split("/")[1] if "/" in e.record_id else "record"
                out[e.key] = Chunk(e.record_id, e.source, SOURCE_NAMES.get(e.source, e.source), section, e.url,
                                   e.text, drug=drug)  # fmt: skip
                n += 1
        return n

    for p in products:
        name = p["name"]
        counts = {}
        ob = runner.call("orange_book", {"drug": name})
        counts["orange_book"] = keep(ob.evidence, name) if ob.ok else 0
        fda = runner.call("drugs_at_fda", {"drug": name})
        counts["drugs_at_fda"] = keep(fda.evidence, name) if fda.ok else 0
        generic = generic_query([i for e in fda.evidence for i in e.fields.get("ingredients", [])])
        counts["ctgov"] = 0
        if generic:
            for status in (ACTIVE, None):  # active Phase 3 first, so readouts are not crowded out by old trials
                args = {"intervention": generic, "phase": "PHASE3", "max_results": 50}
                if status:
                    args["status"] = status
                ct = runner.call("ctgov_search", args)
                counts["ctgov"] += keep(ct.evidence, name) if ct.ok else 0
        log(f"{name} ({generic}): {counts}")
    return list(out.values())


def build_index(cfg: dict, runner: ToolRunner, log=print) -> Path:
    data_dir = Path(cfg.get("data_dir", "data"))
    doc_dir, out_dir = data_dir / "index", data_dir / "index_baseline"
    meta = json.loads((doc_dir / "meta.json").read_text(encoding="utf8"))
    ds = cfg["docsearch"]
    with open(ds["products"], encoding="utf8") as f:
        products = yaml.safe_load(f)["products"]
    embedder = OllamaEmbedder(cfg.get("ollama_url", "http://localhost:11434"), meta["embed_model"])
    docs = Index.load(doc_dir, embedder)
    records = record_chunks(runner, products, log)
    log(f"embedding {len(records)} record chunks")
    rec_emb = embedder.embed([c.search_text for c in records], "document")
    index = Index(docs.chunks + records, np.vstack([docs.emb, rec_emb]), embedder)
    by_source: dict[str, int] = {}
    for c in index.chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    index.save(out_dir, meta | {"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "corpus": "strong",
                                "chunks": len(index.chunks), "by_source": by_source})  # fmt: skip
    log(f"baseline index: {by_source} -> {out_dir}")
    return out_dir


def load_index(cfg: dict, fake: bool = False) -> Index:
    if fake:
        from diligence_agent.fake import fixture

        chunks = [Chunk(**json.loads(x)) for x in fixture("doc_chunks.jsonl").decode("utf8").splitlines()]
        return Index.build(chunks, FakeEmbedder())
    directory = Path(cfg.get("data_dir", "data")) / "index_baseline"
    if not (directory / "chunks.jsonl").exists():
        raise RuntimeError("no baseline index: run `uv run diligence-agent index-baseline` first")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf8"))
    return Index.load(directory, OllamaEmbedder(cfg.get("ollama_url", "http://localhost:11434"), meta["embed_model"]))


@dataclass
class BaselineResult:
    question: str
    answer: str = ""
    retrieved: list[str] = field(default_factory=list)
    records: dict[str, str] = field(default_factory=dict)  # key -> text shown to the model
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    trace_id: str | None = None

    @property
    def cited(self) -> list[str]:
        return citations(self.answer)

    def summary(self) -> dict[str, Any]:
        return {
            "model_calls": 1, "tool_calls": [], "citations": len(self.cited),
            "citations_to_unseen_records": [k for k in self.cited if k not in self.records],
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_usd": self.cost_usd,
            "latency_s": round(self.latency_s, 2), "retrieve_ms": round(self.retrieve_ms),
            "generate_ms": round(self.generate_ms), "trace_id": self.trace_id,
        }  # fmt: skip


def run_question(question: str, index: Index, model, cfg: dict, tags: dict | None = None) -> BaselineResult:
    k = cfg.get("baseline", {}).get("k", 8)
    method = cfg.get("docsearch", {}).get("retriever", "hybrid")
    model_name = cfg.get("chat_model", "gemma4:e4b")
    r = BaselineResult(question=question)
    t_all = time.perf_counter()
    with mlflow.start_span(name="baseline", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"question": question})
        root.set_attributes({"agent": "baseline", "chat_model": model_name, "prompt_sha": PROMPT_SHA, "k": k})
        t0 = time.perf_counter()
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            hits = index.search(question, k=k, method=method)
            r.retrieve_ms = (time.perf_counter() - t0) * 1000
            r.records = {f"{c.source}:{c.chunk_id}": f"{c.doc}, {c.section}: {c.text}" for c, _, _ in hits}
            r.retrieved = list(r.records)
            span.set_inputs({"query": question, "method": method, "k": k})
            span.set_outputs({"keys": r.retrieved})
            span.set_attribute("latency_ms", r.retrieve_ms)
        context = "\n\n".join(f"[{key}] {text}" for key, text in r.records.items())
        system = SYSTEM_PROMPT.format(today=date.today().isoformat(), decline=DECLINE)
        t0 = time.perf_counter()
        with mlflow.start_span(name="generate", span_type=SpanType.CHAT_MODEL) as span:
            msg = model.invoke([SystemMessage(system), HumanMessage(f"Records:\n\n{context}\n\nQuestion: {question}")])
            r.generate_ms = (time.perf_counter() - t0) * 1000
            r.answer = msg.content if isinstance(msg.content, str) else msg.text
            usage = getattr(msg, "usage_metadata", None) or {}
            r.input_tokens, r.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            span.set_outputs({"text": r.answer[:4000]})
            timings = ollama_timings(getattr(msg, "response_metadata", {}) or {})
            span.set_attributes(model_span_attributes(model_name, usage, cfg.get("pricing"), r.generate_ms,
                                                      PROMPT_SHA, timings))  # fmt: skip
        r.cost_usd = cost(cfg.get("pricing"), model_name, r.input_tokens, r.output_tokens)["total_cost"]
        r.latency_s = time.perf_counter() - t_all
        root.set_outputs({"answer": r.answer, "citations": r.cited})
        root.set_attributes({"latency_ms": r.latency_s * 1000, "input_tokens": r.input_tokens,
                             "output_tokens": r.output_tokens, "cost_usd": r.cost_usd})  # fmt: skip
        trace_id = getattr(root, "trace_id", None)
        if trace_id and str(trace_id).startswith("tr-"):
            r.trace_id = trace_id
            mlflow.update_current_trace(tags={k: str(v) for k, v in (tags or {}).items()} | {"agent": "baseline"})
    return r
