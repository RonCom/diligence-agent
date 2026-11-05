"""PubMed E-utilities: esearch for ids, efetch for titles and abstracts."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

from diligence_agent.evidence import Evidence
from diligence_agent.tools.base import Tool, ToolContext

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ABSTRACT_CHARS = 3000


def _text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def parse_efetch(xml: str) -> list[Evidence]:
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = _text(art.find("MedlineCitation/PMID"))
        a = art.find("MedlineCitation/Article")
        title = _text(a.find("ArticleTitle"))
        journal = _text(a.find("Journal/Title"))
        pub = a.find("Journal/JournalIssue/PubDate")
        year = _text(pub.find("Year")) or _text(pub.find("MedlineDate"))[:4]
        parts = []
        for ab in a.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
        abstract = " ".join(parts)
        if len(abstract) > ABSTRACT_CHARS:
            abstract = abstract[:ABSTRACT_CHARS] + " ...[truncated]"
        out.append(
            Evidence(
                source="pubmed",
                record_id=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                text=f"{title} {journal} ({year}). PMID {pmid}. Abstract: {abstract or 'none'}",
                fields={"pmid": pmid, "title": title, "journal": journal, "year": year},
            )
        )
    return out


class PubmedArgs(BaseModel):
    query: str = Field(min_length=2, description="PubMed query, e.g. 'elexacaftor phase 3 randomized'")
    max_results: int = Field(5, ge=1, le=20)


class PubmedTool(Tool):
    name = "pubmed_search"
    description = "Search PubMed and return the top articles with title, journal, year and abstract."
    Args = PubmedArgs

    def run(self, args: PubmedArgs, ctx: ToolContext) -> list[Evidence]:
        base = {"db": "pubmed", "tool": "diligence-agent"}
        found = ctx.http.get(
            EUTILS + "esearch.fcgi",
            params={**base, "term": args.query, "retmode": "json", "retmax": args.max_results, "sort": "relevance"},
        )
        ids = found.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetched = ctx.http.get(EUTILS + "efetch.fcgi", params={**base, "id": ",".join(ids), "retmode": "xml"})
        return parse_efetch(fetched.text)
