"""Hybrid search over the corpus, adapted from policy-rag's retrieve.py.

BM25 matches exact tokens (drug names, "10-K", percentages, years); embeddings match paraphrase. Hybrid fuses the
two rankings with reciprocal rank fusion (k = 60). Rerank re-scores hybrid's top candidates with a cross-encoder,
which reads the query and a chunk together. In policy-rag the reranker was tested and not adopted (+2 points recall
at 5, interval -4 to +8), so hybrid is the default here too, and rerank gets its own test on this eval's dev set.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

from diligence_agent.corpus import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
STOP = set(
    "a an and are as at be by for from has have if in into is it its of on or that the their this to was were which "
    "with not no may can will shall should must our we".split()
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.len = np.array([len(t) for t in toks], float)
        self.avg = self.len.mean() if len(toks) else 0.0
        df = Counter(w for t in toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [Counter(t) for t in toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return s


# ---- embeddings ----------------------------------------------------------------------------------------------


class OllamaEmbedder:
    def __init__(self, url: str, model: str, timeout_s: float = 600):
        self.url, self.model = url.rstrip("/"), model
        self.client = httpx.Client(timeout=timeout_s)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        # nomic-embed-text was trained with task prefixes; other models ignore them
        prefix = {"query": "search_query: ", "document": "search_document: "}.get(kind, "")
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            r = self.client.post(
                f"{self.url}/api/embed", json={"model": self.model, "input": [prefix + t for t in texts[i : i + 32]]}
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeEmbedder:
    """Hashed bag of words: deterministic, no model server."""

    model = "fake-hash-256"
    dim = 256

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        v = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                v[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


# ---- rerankers ----------------------------------------------------------------------------------------------

_CROSS_ENCODERS: dict = {}


class CrossEncoderReranker:
    def __init__(self, model: str, max_length: int = 512):
        if model not in _CROSS_ENCODERS:
            try:
                from sentence_transformers import CrossEncoder  # torch loads only when rerank is used
            except ImportError as exc:
                from diligence_agent.tools.base import ToolConfigError

                raise ToolConfigError("retriever 'rerank' needs the extra: uv sync --extra rerank") from exc

            _CROSS_ENCODERS[model] = CrossEncoder(model, max_length=max_length)
        self.model = _CROSS_ENCODERS[model]

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        pairs = [(query, t) for t in texts]
        return np.asarray(self.model.predict(pairs, batch_size=16, show_progress_bar=False), dtype=float)


class FakeReranker:
    """Share of query tokens found in the chunk; stands in for the cross-encoder in tests and --fake runs."""

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) / (len(q) or 1) for t in texts], dtype=float)


# ---- index ----------------------------------------------------------------------------------------------------


class Index:
    def __init__(self, chunks: list[Chunk], emb: np.ndarray, embedder, reranker_factory=None, candidates: int = 20):
        self.chunks = chunks
        self.emb = emb
        self.embedder = embedder
        self.docs = [c.search_text for c in chunks]
        self.lower = [d.casefold() for d in self.docs]
        self.bm25 = BM25(self.docs)
        self.candidates = candidates
        self._reranker_factory = reranker_factory or FakeReranker
        self._reranker = None

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, **kw) -> Index:
        emb = embedder.embed([c.search_text for c in chunks], "document") if chunks else np.zeros((0, 1), np.float32)
        return cls(chunks, emb, embedder, **kw)

    def save(self, directory: Path, meta: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "chunks.jsonl", "w", encoding="utf8", newline="\n") as f:
            for c in self.chunks:
                f.write(json.dumps(c.as_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.emb)
        (directory / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")

    @classmethod
    def load(cls, directory: Path, embedder, **kw) -> Index:
        with open(directory / "chunks.jsonl", encoding="utf8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(directory / "embeddings.npy"), embedder, **kw)

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = self._reranker_factory()
        return self._reranker

    def _scores(self, query: str, method: str) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(query)
        if method == "dense":
            return self.emb @ self.embedder.embed([query], "query")[0]
        raise ValueError(method)

    def _rrf(self, query: str, mask: np.ndarray) -> np.ndarray:
        s = np.zeros(len(self.chunks))
        for m in ("bm25", "dense"):
            sc = np.where(mask, self._scores(query, m), -np.inf)
            order = np.argsort(-sc, kind="stable")
            s[order] += 1.0 / (RRF_K + np.arange(1, len(order) + 1))
        return np.where(mask, s, -np.inf)

    def mask(self, source: str | None = None, mentions: str | None = None) -> np.ndarray:
        m = np.ones(len(self.chunks), bool)
        if source:
            m &= np.array([c.source == source for c in self.chunks])
        if mentions:
            q = mentions.casefold()
            m &= np.array(
                [(c.drug or "").casefold() == q or q in low for c, low in zip(self.chunks, self.lower, strict=True)]
            )
        return m

    def search(self, query: str, k: int = 5, method: str = "hybrid", mask: np.ndarray | None = None):
        """[(chunk, score, rank)] best first."""
        mask = np.ones(len(self.chunks), bool) if mask is None else mask
        if not mask.any():
            return []
        if method in ("bm25", "dense"):
            s = np.where(mask, self._scores(query, method), -np.inf)
        elif method == "hybrid":
            s = self._rrf(query, mask)
        elif method == "rerank":
            fused = self._rrf(query, mask)
            cand = [i for i in np.argsort(-fused, kind="stable")[: max(self.candidates, k)] if mask[i]]
            s = np.full(len(self.chunks), -np.inf)
            s[cand] = self.reranker.scores(query, [self.docs[i] for i in cand])
        else:
            raise ValueError(method)
        top = [i for i in np.argsort(-s, kind="stable")[:k] if np.isfinite(s[i])]
        return [(self.chunks[i], float(s[i]), r) for r, i in enumerate(top, 1)]
