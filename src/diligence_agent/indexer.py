"""Build the doc_search index: latest filings from config.yaml plus the current FDA label for each product."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from diligence_agent.corpus import Chunk, filing_chunks, label_chunks, slug
from diligence_agent.retrieval import Index, OllamaEmbedder
from diligence_agent.tools.base import ToolRunner
from diligence_agent.tools.edgar import sec_headers
from diligence_agent.tools.openfda import DAILYMED, LABEL_API, _quoted, _search


def filing_corpus(runner: ToolRunner, filings: list[dict], words: int, overlap: int, log=print) -> list[Chunk]:
    ctx = runner.ctx
    out: list[Chunk] = []
    for f in filings:
        listed = runner.call("edgar_filings", {"company": f["company"], "form": f.get("form", "10-K"), "limit": 1})
        if not listed.ok or not listed.evidence:
            raise RuntimeError(f"no {f.get('form', '10-K')} for {f['company']}: {listed.error or 'none listed'}")
        e = listed.evidence[0]
        url = e.fields["document_url"]
        dest = Path(ctx.data_dir) / "raw" / "edgar" / url.rsplit("/", 1)[-1]
        html = ctx.http.download(url, dest, headers=sec_headers(ctx)).read_bytes()
        period = e.fields["period"] or e.fields["filed"]
        key = f"{f['company'].lower()}-{e.fields['form'].lower()}-{period[:4]}"
        doc = f"{e.text.split(' (CIK')[0]} {e.fields['form']}, period ending {period}"
        chunks = filing_chunks(html, key, doc, url, words, overlap)
        log(f"{doc}: {len(chunks)} chunks")
        out.extend(chunks)
    return out


def pick_label(results: list[dict], name: str) -> dict | None:
    """Newest label whose brand name matches; openFDA also returns repackager and co-marketed labels."""
    want = name.casefold().split()[0]

    def names(r: dict) -> list[str]:
        return r.get("openfda", {}).get("brand_name", []) + [
            s.split()[0] for s in r.get("spl_product_data_elements", []) if s.split()
        ]

    named = [r for r in results if any(want in n.casefold() for n in names(r))]
    pool = named or results
    return max(pool, key=lambda r: r.get("effective_time", ""), default=None)


def label_corpus(runner: ToolRunner, products: list[dict], words: int, overlap: int, log=print) -> list[Chunk]:
    ctx = runner.ctx
    out: list[Chunk] = []
    for p in products:
        name = p["name"]
        variants = list(dict.fromkeys([name, name.split()[0]]))
        queries = [f"openfda.brand_name:{_quoted(n)}" for n in variants]
        queries += [f"spl_product_data_elements:{_quoted(n)}" for n in variants]
        label = pick_label(_search(ctx, LABEL_API, queries, limit=10), name)
        if label is None:
            log(f"{name}: no label found")
            continue
        dest = Path(ctx.data_dir) / "raw" / "labels" / f"{slug(name)}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(label), encoding="utf8")
        chunks = label_chunks(label, name, DAILYMED.format(label["set_id"]), words, overlap)
        log(f"{name}: label {label.get('effective_time')} ({len(chunks)} chunks)")
        out.extend(chunks)
    return out


def ollama_digest(url: str, model: str) -> str | None:
    try:
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=10).json()["models"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    full = model if ":" in model else f"{model}:latest"
    return next((m["digest"][:12] for m in tags if m["name"] == full), None)


def build_index(cfg: dict, runner: ToolRunner, log=print) -> Path:
    ds = cfg["docsearch"]
    words, overlap = ds.get("chunk_words", 300), ds.get("overlap_words", 40)
    with open(ds["products"], encoding="utf8") as f:
        drug_set = yaml.safe_load(f)
    if drug_set.get("status") != "confirmed":
        raise RuntimeError(f"{ds['products']} is not confirmed")
    start = time.perf_counter()
    chunks = filing_corpus(runner, ds["filings"], words, overlap, log) + label_corpus(
        runner, drug_set["products"], words, overlap, log
    )
    url = cfg.get("ollama_url", "http://localhost:11434")
    embedder = OllamaEmbedder(url, ds["embed_model"])
    log(f"embedding {len(chunks)} chunks with {ds['embed_model']}")
    index = Index.build(chunks, embedder)
    directory = Path(cfg.get("data_dir", "data")) / "index"
    index.save(
        directory,
        {
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "embed_model": ds["embed_model"],
            "embed_model_digest": ollama_digest(url, ds["embed_model"]),
            "chunk_words": words,
            "overlap_words": overlap,
            "chunks": len(chunks),
            "by_source": {s: sum(c.source == s for c in chunks) for s in ("sec_filing", "label_text")},
            "seconds": round(time.perf_counter() - start, 1),
        },
    )
    log(f"index written to {directory} in {time.perf_counter() - start:.0f}s")
    return directory
