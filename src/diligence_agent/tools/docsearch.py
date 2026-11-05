"""Search over downloaded SEC filings and full FDA label text (built by `diligence-agent index`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from diligence_agent.evidence import Evidence
from diligence_agent.retrieval import CrossEncoderReranker, FakeEmbedder, FakeReranker, Index, OllamaEmbedder
from diligence_agent.tools.base import Tool, ToolConfigError, ToolContext

SOURCES = {"10-K": "sec_filing", "label": "label_text"}


def load_index(ctx: ToolContext) -> Index:
    if "doc_index" in ctx.memo:
        return ctx.memo["doc_index"]
    ds = ctx.settings.get("docsearch", {})
    candidates = ds.get("rerank_candidates", 20)
    if ctx.fake:
        from diligence_agent.corpus import Chunk
        from diligence_agent.fake import fixture

        chunks = [Chunk(**json.loads(line)) for line in fixture("doc_chunks.jsonl").decode("utf8").splitlines()]
        index = Index.build(chunks, FakeEmbedder(), reranker_factory=FakeReranker, candidates=candidates)
    else:
        directory = Path(ctx.data_dir) / "index"
        if not (directory / "chunks.jsonl").exists():
            raise ToolConfigError("no search index: run `uv run diligence-agent index` first")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf8"))
        embedder = OllamaEmbedder(ctx.settings.get("ollama_url", "http://localhost:11434"), meta["embed_model"])
        model = ds.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        index = Index.load(
            directory, embedder, reranker_factory=lambda: CrossEncoderReranker(model), candidates=candidates
        )
    ctx.memo["doc_index"] = index
    return index


class DocSearchArgs(BaseModel):
    query: str = Field(min_length=2, description="What to look for, e.g. 'Xtandi royalty rate and duration'")
    drug: str | None = Field(None, description="Keep only passages about this product (label or mentioned by name)")
    source: Literal["10-K", "label"] | None = Field(None, description="Search only 10-K filings or only FDA labels")
    k: int = Field(5, ge=1, le=10, description="Number of passages")


class DocSearchTool(Tool):
    name = "doc_search"
    description = (
        "Search the full text of Royalty Pharma's latest 10-K and the FDA labels of its royalty products. Use it for "
        "royalty rates and durations, 2025 receipts and end-market sales, deal terms, and label details such as "
        "clinical study results. Returns passages with their section."
    )
    Args = DocSearchArgs
    cacheable = False  # milliseconds to run, and results change when the index is rebuilt

    def run(self, args: DocSearchArgs, ctx: ToolContext) -> list[Evidence]:
        index = load_index(ctx)
        method = ctx.settings.get("docsearch", {}).get("retriever", "hybrid")
        mask = index.mask(source=SOURCES.get(args.source) if args.source else None, mentions=args.drug)
        return [
            Evidence(
                source=c.source,
                record_id=c.chunk_id,
                url=c.url,
                text=f"{c.doc}, {c.section}: {c.text}",
                fields={
                    "doc": c.doc,
                    "section": c.section,
                    "drug": c.drug,
                    "rank": rank,
                    "score": score,
                    "retriever": method,
                },
            )
            for c, score, rank in index.search(args.query, k=args.k, method=method, mask=mask)
        ]
