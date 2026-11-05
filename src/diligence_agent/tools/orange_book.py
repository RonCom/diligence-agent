"""FDA Orange Book data files: products, patents and exclusivity for NDA drugs.

The files list one row per product strength, and the patent file one row per patent x product x use code; Trikafta's
two applications alone have 289 patent rows. Records here collapse those to one per application, one per
(application, patent, expiry) and one per (application, exclusivity code, expiry).
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from diligence_agent.evidence import Evidence
from diligence_agent.tools.base import Tool, ToolContext

OB_URL = "https://www.fda.gov/media/76860/download?attachment"
OB_PAGE = "https://www.accessdata.fda.gov/scripts/cder/ob/results_product.cfm?Appl_Type={t}&Appl_No={n}"

EXCLUSIVITY_PREFIX = {
    "NCE": "new chemical entity",
    "ODE": "orphan drug",
    "PED": "pediatric",
    "NP": "new product",
    "NPP": "new patient population",
    "I": "new indication",
    "D": "new dosing schedule",
    "M": "miscellaneous",
    "RTO": "change from prescription to OTC",
    "CGT": "competitive generic therapy",
    "GAIN": "qualified infectious disease product",
}


def parse_date(text: str) -> str | None:
    text = (text or "").strip()
    for fmt in ("%b %d, %Y", "%b %d,%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def exclusivity_meaning(code: str) -> str:
    prefix = code.split("-")[0].rstrip("*")
    return EXCLUSIVITY_PREFIX.get(prefix, "see the Orange Book exclusivity code list")


class OrangeBookData:
    def __init__(self, products: pd.DataFrame, patents: pd.DataFrame, exclusivity: pd.DataFrame) -> None:
        self.products, self.patents, self.exclusivity = products, patents, exclusivity

    @classmethod
    def from_zip(cls, blob: bytes) -> OrangeBookData:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = {n.lower(): n for n in z.namelist()}

            def read(name: str) -> pd.DataFrame:
                raw = z.read(names[name]).decode("latin-1")
                return pd.read_csv(io.StringIO(raw), sep="~", dtype=str, keep_default_na=False)

            return cls(read("products.txt"), read("patent.txt"), read("exclusivity.txt"))

    @classmethod
    def load(cls, ctx: ToolContext) -> OrangeBookData:
        if "orange_book" not in ctx.memo:
            if ctx.fake:
                from diligence_agent.fake import orange_book_zip

                blob = orange_book_zip()
            else:
                blob = ctx.http.download(OB_URL, ctx.data_dir / "raw" / "orange_book.zip").read_bytes()
            ctx.memo["orange_book"] = cls.from_zip(blob)
        return ctx.memo["orange_book"]

    def match(self, drug: str) -> pd.DataFrame:
        """Exact trade name, then trade name before a qualifier ("TRIKAFTA (COPACKAGED)"), then ingredient."""
        q = " ".join(drug.split()).upper()
        p = self.products
        trade = p["Trade_Name"].str.upper()
        base = trade.str.replace(r"\s*\(.*\)$", "", regex=True)
        for mask in (trade == q, base == q, p["Ingredient"].str.upper() == q):
            if mask.any():
                return p[mask]
        return p[p["Ingredient"].str.upper().str.contains(q, regex=False)]


class OrangeBookArgs(BaseModel):
    drug: str = Field(min_length=2, description="Brand (trade) name or active ingredient, e.g. TRIKAFTA or ivacaftor")


class OrangeBookTool(Tool):
    name = "orange_book"
    description = (
        "FDA Orange Book records for a small-molecule drug approved under an NDA: each application with its approval "
        "date and applicant, every listed patent with its expiration date, and every regulatory exclusivity with its "
        "expiration date. Biologics (BLAs) are not in the Orange Book."
    )
    Args = OrangeBookArgs
    version = 3  # 2: latest expiries per application and drug; 3: drug key without spaces

    def run(self, args: OrangeBookArgs, ctx: ToolContext) -> list[Evidence]:
        ob = OrangeBookData.load(ctx)
        prods = ob.match(args.drug)
        if prods.empty:
            return []
        out: list[Evidence] = []
        apps: list[Evidence] = []
        for (atype, ano), g in prods.groupby(["Appl_Type", "Appl_No"], sort=True):
            trade = g["Trade_Name"].iloc[0]
            url = OB_PAGE.format(t=atype, n=ano)
            details = self._patents_and_exclusivity(ob, atype, ano, trade, url)
            approvals = sorted({parse_date(d) for d in g["Approval_Date"] if parse_date(d)})
            strengths = "; ".join(
                f"product {r['Product_No']}: {r['Strength']} {r['DF;Route']}" for _, r in g.iterrows()
            )
            latest_pat = _latest(details, "patent_no")
            latest_excl = _latest(details, "code")
            app = Evidence(
                source="orange_book",
                record_id=f"{atype}{ano}",
                url=url,
                text=(
                    f"{trade} ({g['Ingredient'].iloc[0]}), application {atype}{ano}, applicant "
                    f"{g['Applicant_Full_Name'].iloc[0]}. Products: {strengths}. Product approval dates: "
                    f"{', '.join(approvals) or g['Approval_Date'].iloc[0]}. "
                    f"{_latest_text('patent', latest_pat, 'patent_no')} "
                    f"{_latest_text('exclusivity', latest_excl, 'code')}"
                ),
                fields={
                    "trade_name": trade,
                    "appl": f"{atype}{ano}",
                    "approval_dates": approvals,
                    "latest_patent_expiry": latest_pat.fields["expires"] if latest_pat else None,
                    "latest_patent_no": latest_pat.fields["patent_no"] if latest_pat else None,
                    "latest_exclusivity_expiry": latest_excl.fields["expires"] if latest_excl else None,
                    "latest_exclusivity_code": latest_excl.fields["code"] if latest_excl else None,
                    "patents_listed": sum("patent_no" in e.fields for e in details),
                },
            )
            apps.append(app)
            out.append(app)
            out.extend(details)
        if len(apps) > 1:
            out.insert(0, _drug_summary(args.drug, apps, out))
        return out

    @staticmethod
    def _patents_and_exclusivity(ob: OrangeBookData, atype: str, ano: str, trade: str, url: str) -> list[Evidence]:
        """Patent and exclusivity records for one application, latest expiry first."""
        out: list[Evidence] = []
        pats = ob.patents[(ob.patents["Appl_Type"] == atype) & (ob.patents["Appl_No"] == ano)]
        for (pno, exp), pg in pats.groupby(["Patent_No", "Patent_Expire_Date_Text"], sort=False):
            expires = parse_date(exp)
            uses = sorted({u for u in pg["Patent_Use_Code"] if u})
            claims = [
                k
                for k, col in (("drug substance", "Drug_Substance_Flag"), ("drug product", "Drug_Product_Flag"))
                if (pg[col] == "Y").any()
            ]
            out.append(
                Evidence(
                    source="orange_book",
                    record_id=f"{atype}{ano}/patent/{pno}",
                    url=url,
                    text=(
                        f"Patent {pno} is listed for {trade} ({atype}{ano}, products "
                        f"{', '.join(sorted(set(pg['Product_No'])))}) and expires {expires or exp}. "
                        f"Claims: {', '.join(claims) or 'method of use only'}. "
                        f"Use codes: {', '.join(uses) or 'none'}."
                        + (" Delisted." if (pg["Delist_Flag"] == "Y").any() else "")
                    ),
                    fields={
                        "appl": f"{atype}{ano}",
                        "patent_no": pno,
                        "expires": expires,
                        "pediatric_extension": pno.endswith("*PED"),
                        "use_codes": uses,
                        "delisted": bool((pg["Delist_Flag"] == "Y").any()),
                    },
                )
            )
        excl = ob.exclusivity[(ob.exclusivity["Appl_Type"] == atype) & (ob.exclusivity["Appl_No"] == ano)]
        for (code, exp), eg in excl.groupby(["Exclusivity_Code", "Exclusivity_Date"], sort=False):
            expires = parse_date(exp)
            out.append(
                Evidence(
                    source="orange_book",
                    record_id=f"{atype}{ano}/exclusivity/{code}/{expires}",
                    url=url,
                    text=(
                        f"Exclusivity {code} ({exclusivity_meaning(code)}) for {trade} ({atype}{ano}, products "
                        f"{', '.join(sorted(set(eg['Product_No'])))}) expires {expires or exp}."
                    ),
                    fields={"appl": f"{atype}{ano}", "code": code, "expires": expires},
                )
            )
        out.sort(key=lambda e: e.fields.get("expires") or "", reverse=True)
        return out


def _latest(records: list[Evidence], kind: str) -> Evidence | None:
    """The record of one kind ("patent_no" or "code") with the latest expiry; delisted patents don't count."""
    dated = [e for e in records if kind in e.fields and e.fields.get("expires") and not e.fields.get("delisted")]
    return max(dated, key=lambda e: e.fields["expires"], default=None)


def _latest_text(label: str, rec: Evidence | None, kind: str) -> str:
    if rec is None:
        return f"No {label} listed."
    extra = " (includes pediatric exclusivity)" if rec.fields.get("pediatric_extension") else ""
    return f"Latest-expiring {label}: {rec.fields[kind]}, expires {rec.fields['expires']}{extra}."


def _drug_summary(query: str, apps: list[Evidence], records: list[Evidence]) -> Evidence:
    """Across all applications that match the drug, e.g. Trikafta's tablet and granule NDAs."""
    pat = _latest(records, "patent_no")
    excl = _latest(records, "code")
    names = ", ".join(f"{a.fields['appl']} ({a.fields['trade_name']})" for a in apps)
    return Evidence(
        source="orange_book",
        record_id=f"drug/{re.sub(r'[^A-Z0-9]+', '-', query.upper()).strip('-')}",
        url=apps[0].url,
        text=(
            f"Across {len(apps)} applications matching {query!r} ({names}): "
            + (
                f"latest-expiring patent {pat.fields['patent_no']} ({pat.fields['appl']}), "
                f"expires {pat.fields['expires']}. "
                if pat
                else "no patents listed. "
            )
            + (
                f"Latest-expiring exclusivity {excl.fields['code']} ({excl.fields['appl']}), "
                f"expires {excl.fields['expires']}."
                if excl
                else "No exclusivity listed."
            )
        ),
        fields={
            "applications": [a.fields["appl"] for a in apps],
            "latest_patent_expiry": pat.fields["expires"] if pat else None,
            "latest_patent_no": pat.fields["patent_no"] if pat else None,
            "latest_exclusivity_expiry": excl.fields["expires"] if excl else None,
            "latest_exclusivity_code": excl.fields["code"] if excl else None,
        },
    )
