"""doc_search over the --fake index: real 10-K and label chunks, hashed embeddings, token-overlap reranker."""

from __future__ import annotations

import numpy as np
import pytest

from diligence_agent.corpus import Chunk
from diligence_agent.retrieval import BM25, FakeEmbedder, Index


def test_finds_royalty_table_row(runner):
    out = runner.call("doc_search", {"query": "Xtandi royalty rate estimated duration", "source": "10-K", "k": 3})
    assert out.ok, out.error
    assert any("Xtandi | 2016 | 2027-2028 | Slightly less than 4% royalty" in e.text for e in out.evidence)
    assert all(e.source == "sec_filing" for e in out.evidence)
    assert [e.fields["rank"] for e in out.evidence] == [1, 2, 3]


def test_drug_and_source_filters(runner):
    out = runner.call("doc_search", {"query": "indications", "drug": "Trikafta", "source": "label"})
    assert out.evidence and all(e.fields["drug"] == "Trikafta" for e in out.evidence)
    assert runner.call("doc_search", {"query": "x y", "drug": "Keytruda"}).evidence == []


def test_not_cached(runner, tmp_path):
    from diligence_agent.cache import ToolCache
    from diligence_agent.tools.base import ToolRunner

    r = ToolRunner(list(runner.tools.values()), runner.ctx, cache=ToolCache(tmp_path / "c"))
    r.call("doc_search", {"query": "Tysabri sales"})
    assert not r.call("doc_search", {"query": "Tysabri sales"}).cache_hit


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid", "rerank"])
def test_every_retriever_ranks_the_relevant_chunk_first(method):
    chunks = [
        Chunk("a", "sec_filing", "d", "Portfolio Summary", "u", "Xtandi | 2016 | 2027-2028 | Slightly less than 4%"),
        Chunk("b", "sec_filing", "d", "Risk Factors", "u", "Interest rates may rise and borrowing costs increase."),
        Chunk("c", "label_text", "d", "Indications", "u", "TRIKAFTA treats cystic fibrosis", drug="Trikafta"),
    ]
    index = Index.build(chunks, FakeEmbedder())
    top = index.search("Xtandi royalty 2027-2028", k=2, method=method)
    assert top[0][0].chunk_id == "a"
    masked = index.search("Xtandi", k=3, method=method, mask=index.mask(source="label_text"))
    assert [c.chunk_id for c, _, _ in masked] == ["c"]


def test_index_round_trip(tmp_path):
    chunks = [Chunk("a", "sec_filing", "d", "S", "u", "alpha beta"), Chunk("b", "sec_filing", "d", "S", "u", "gamma")]
    index = Index.build(chunks, FakeEmbedder())
    index.save(tmp_path, {"embed_model": "fake"})
    again = Index.load(tmp_path, FakeEmbedder())
    assert [c.chunk_id for c in again.chunks] == ["a", "b"]
    assert np.allclose(again.emb, index.emb)


def test_bm25_prefers_rare_terms():
    bm = BM25(["royalty royalty royalty", "royalty xtandi", "other words"])
    assert int(np.argmax(bm.scores("xtandi royalty"))) == 1
