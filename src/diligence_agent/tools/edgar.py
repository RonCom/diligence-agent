"""SEC EDGAR filing lists from the submissions JSON. SEC requires a declared User-Agent with a name and email."""

from __future__ import annotations

from pydantic import BaseModel, Field

from diligence_agent.evidence import Evidence
from diligence_agent.tools.base import Tool, ToolConfigError, ToolContext

TICKERS = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def sec_headers(ctx: ToolContext) -> dict[str, str]:
    ua = ctx.env.get("SEC_USER_AGENT", "").strip()
    if ctx.fake:
        return {}
    if "@" not in ua:
        raise ToolConfigError(
            "SEC_USER_AGENT is not set in .env (SEC needs a name and email, e.g. 'Jane Doe jd@x.com')"
        )
    return {"User-Agent": ua}


def resolve_cik(company: str, ctx: ToolContext) -> tuple[int, str] | None:
    company = company.strip()
    if company.isdigit():
        return int(company), company
    if "sec_tickers" not in ctx.memo:
        ctx.memo["sec_tickers"] = list(ctx.http.get(TICKERS, headers=sec_headers(ctx)).json().values())
    q = company.casefold()
    rows = ctx.memo["sec_tickers"]
    for test in (
        lambda r: r["ticker"].casefold() == q,
        lambda r: r["title"].casefold() == q,
        lambda r: q in r["title"].casefold(),
    ):
        hits = [r for r in rows if test(r)]
        if hits:
            return int(hits[0]["cik_str"]), hits[0]["title"]
    return None


class FilingsArgs(BaseModel):
    company: str = Field(min_length=1, description="Ticker, company name or CIK, e.g. RPRX")
    form: str = Field("10-K", description="Form type, e.g. 10-K, 10-Q, 8-K")
    limit: int = Field(3, ge=1, le=10)


class EdgarFilingsTool(Tool):
    name = "edgar_filings"
    description = (
        "List a company's recent SEC filings of one form type, newest first, with filing date, period and a link to "
        "the main document. Use the 10-K for annual sales, royalty terms and patent disclosures."
    )
    Args = FilingsArgs

    def run(self, args: FilingsArgs, ctx: ToolContext) -> list[Evidence]:
        found = resolve_cik(args.company, ctx)
        if found is None:
            return []
        cik, title = found
        resp = ctx.http.get(SUBMISSIONS.format(cik), headers=sec_headers(ctx))
        if resp.status_code == 404:
            return []
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        name = data.get("name") or title
        form = args.form.strip().upper()
        out = []
        for i, f in enumerate(recent.get("form", [])):
            if f.upper() != form:
                continue
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            url = ARCHIVE.format(cik=cik, acc=acc.replace("-", ""), doc=doc)
            period, filed = recent.get("reportDate", [""] * (i + 1))[i], recent["filingDate"][i]
            out.append(
                Evidence(
                    source="edgar",
                    record_id=acc,
                    url=url,
                    text=(
                        f"{name} (CIK {cik}) filed a {f} for the period ending {period or 'n/a'} on {filed}, "
                        f"accession {acc}."
                    ),
                    fields={
                        "cik": cik,
                        "form": f,
                        "accession": acc,
                        "filed": filed,
                        "period": period,
                        "document_url": url,
                    },
                )
            )
            if len(out) == args.limit:
                break
        return out
