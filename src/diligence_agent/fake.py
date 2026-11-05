"""--fake mode: an httpx transport that serves recorded responses, so tools, agents and tests run offline.

The FDA, ClinicalTrials.gov and PubMed fixtures are real responses for Trikafta (elexacaftor/tezacaftor/ivacaftor),
recorded 2026-11-04 and trimmed. Queries about any other drug get each source's own "no match" reply. The SEC
fixtures describe a made-up company (EXPH, CIK 1234567) until a recorded filing list replaces them.
"""

from __future__ import annotations

import io
import json
import zipfile
from importlib import resources

import httpx

KNOWN = ("trikafta", "elexacaftor", "tezacaftor", "ivacaftor")
FAKE_CIK = 1234567


def fixture(name: str) -> bytes:
    return resources.files("diligence_agent").joinpath("fixtures", name).read_bytes()


def orange_book_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in ("products.txt", "patent.txt", "exclusivity.txt"):
            z.writestr(name, fixture(f"ob_{name}"))
    return buf.getvalue()


def _known(text: str) -> bool:
    text = text.casefold()
    return any(k in text for k in KNOWN)


def _json(data) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(data).encode(), headers={"content-type": "application/json"})


def handler(request: httpx.Request) -> httpx.Response:
    host, path, q = request.url.host, request.url.path, request.url.params
    not_found = httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}})
    if host == "www.fda.gov":
        return httpx.Response(200, content=orange_book_zip())
    if host == "clinicaltrials.gov" and path == "/api/v2/studies":
        return (
            httpx.Response(200, content=fixture("ctgov_studies.json"))
            if _known(q.get("query.intr", ""))
            else _json({"studies": []})
        )
    if host == "api.fda.gov":
        name = {"/drug/label.json": "openfda_label.json", "/drug/drugsfda.json": "openfda_drugsfda.json"}.get(path)
        return httpx.Response(200, content=fixture(name)) if name and _known(q.get("search", "")) else not_found
    if host == "www.sec.gov" and path == "/files/company_tickers.json":
        return httpx.Response(200, content=fixture("sec_company_tickers.json"))
    if host == "data.sec.gov" and path == f"/submissions/CIK{FAKE_CIK:010d}.json":
        return httpx.Response(200, content=fixture("sec_submissions.json"))
    if host == "eutils.ncbi.nlm.nih.gov":
        if path.endswith("esearch.fcgi"):
            return (
                httpx.Response(200, content=fixture("pubmed_esearch.json"))
                if _known(q.get("term", ""))
                else _json({"esearchresult": {"count": "0", "idlist": []}})
            )
        if path.endswith("efetch.fcgi"):
            return httpx.Response(200, content=fixture("pubmed_efetch.xml"))
    return httpx.Response(404, text="not in fake fixtures")


def fake_transport() -> httpx.MockTransport:
    return httpx.MockTransport(handler)
