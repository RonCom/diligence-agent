"""openFDA drug labels and Drugs@FDA approvals. openFDA answers HTTP 404 when nothing matches."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from diligence_agent.evidence import Evidence
from diligence_agent.tools.base import Tool, ToolContext

LABEL_API = "https://api.fda.gov/drug/label.json"
DRUGSFDA_API = "https://api.fda.gov/drug/drugsfda.json"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={}"
DRUGSFDA_PAGE = "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={}"
SECTION_CHARS = 6000

LabelSection = Literal[
    "indications_and_usage", "boxed_warning", "dosage_and_administration", "contraindications", "warnings_and_cautions"
]


def _quoted(name: str) -> str:
    return '"' + " ".join(name.replace('"', " ").split()) + '"'


def _iso(yyyymmdd: str | None) -> str | None:
    if not yyyymmdd or len(yyyymmdd) != 8:
        return None
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def _search(ctx: ToolContext, url: str, queries: list[str], limit: int) -> list[dict]:
    params = {"limit": limit}
    if key := ctx.env.get("OPENFDA_API_KEY"):
        params["api_key"] = key
    for q in queries:
        resp = ctx.http.get(url, params={**params, "search": q})
        if resp.status_code == 200 and (results := resp.json().get("results")):
            return results
    return []


class LabelArgs(BaseModel):
    drug: str = Field(min_length=2, description="Brand or generic name")
    sections: list[LabelSection] = Field(
        default_factory=lambda: ["indications_and_usage"], description="Label sections to return"
    )


class LabelTool(Tool):
    name = "fda_label"
    description = (
        "Current FDA prescribing information (structured product label) for a drug, by brand or generic name. "
        "Returns the requested sections; indications_and_usage lists the approved indications."
    )
    Args = LabelArgs
    version = 2  # 2: generic name in the record text

    def run(self, args: LabelArgs, ctx: ToolContext) -> list[Evidence]:
        q = _quoted(args.drug)
        # openFDA leaves the harmonized openfda fields empty on some labels (Tysabri, Tremfya, Evrysdi in 2026);
        # spl_product_data_elements, the label's own product listing, still names the product
        queries = [f"openfda.brand_name:{q}", f"openfda.generic_name:{q}", f"spl_product_data_elements:{q}"]
        results = _search(ctx, LABEL_API, queries, limit=5)
        newest: dict[str, dict] = {}
        for r in results:  # one label per set id, newest version
            sid = r.get("set_id", "")
            if sid not in newest or r.get("effective_time", "") > newest[sid].get("effective_time", ""):
                newest[sid] = r
        labels = sorted(newest.values(), key=lambda r: r.get("effective_time", ""), reverse=True)[:2]
        out = []
        for r in labels:
            ofda = r.get("openfda", {})
            brand = ", ".join(ofda.get("brand_name", [])) or args.drug
            generic = ", ".join(ofda.get("generic_name", []))
            named = f"{brand} ({generic.lower()})" if generic else brand
            effective = _iso(r.get("effective_time"))
            for section in args.sections:
                body = " ".join(r.get(section, []))
                if not body:
                    continue
                text = body if len(body) <= SECTION_CHARS else body[:SECTION_CHARS] + " ...[truncated]"
                out.append(
                    Evidence(
                        source="fda_label",
                        record_id=f"{r['set_id']}/{section}@{r.get('effective_time', '')}",
                        url=DAILYMED.format(r["set_id"]),
                        text=f"{named} label ({section.replace('_', ' ')}, effective {effective}): {text}",
                        fields={
                            "brand": brand,
                            "set_id": r["set_id"],
                            "effective": effective,
                            "section": section,
                            "application_number": ofda.get("application_number", []),
                            "manufacturer": ofda.get("manufacturer_name", []),
                        },
                    )
                )
        return out


class DrugsAtFdaArgs(BaseModel):
    drug: str = Field(min_length=2, description="Brand or generic name")


class DrugsAtFdaTool(Tool):
    name = "drugs_at_fda"
    description = (
        "Drugs@FDA approval history for a drug: each NDA or BLA with its sponsor and original approval date, plus "
        "approved efficacy supplements (new indications or populations) with their dates."
    )
    Args = DrugsAtFdaArgs
    version = 2  # 2: ingredients field

    def run(self, args: DrugsAtFdaArgs, ctx: ToolContext) -> list[Evidence]:
        q = _quoted(args.drug)
        apps = _search(
            ctx,
            DRUGSFDA_API,
            [f"openfda.brand_name:{q}", f"products.brand_name:{q}", f"openfda.generic_name:{q}"],
            limit=10,
        )
        out = []
        for app in apps:
            appno = app.get("application_number", "")
            url = DRUGSFDA_PAGE.format("".join(c for c in appno if c.isdigit()))
            brands = sorted({p.get("brand_name", "") for p in app.get("products", [])})
            ingredients = sorted(
                {i.get("name", "") for p in app.get("products", []) for i in p.get("active_ingredients", [])}
            )
            label = f"{', '.join(brands)} ({'; '.join(ingredients)})"
            for s in sorted(app.get("submissions", []), key=lambda s: s.get("submission_status_date", "")):
                if s.get("submission_status") != "AP":
                    continue
                stype, snum, cls = s.get("submission_type"), s.get("submission_number"), s.get("submission_class_code")
                when = _iso(s.get("submission_status_date"))
                if stype == "ORIG":
                    text = f"{label}, {appno}, sponsor {app.get('sponsor_name', '')}: original approval {when}."
                elif cls == "EFFICACY":
                    desc = s.get("submission_class_code_description", "efficacy")
                    text = f"{label}, {appno}: efficacy supplement {snum} ({desc}) approved {when}."
                else:
                    continue
                out.append(
                    Evidence(
                        source="drugs_at_fda",
                        record_id=f"{appno}/{stype}-{snum}",
                        url=url,
                        text=text,
                        fields={
                            "application": appno,
                            "submission": f"{stype}-{snum}",
                            "class": cls,
                            "approved": when,
                            "brands": brands,
                            "ingredients": ingredients,
                        },
                    )
                )
        return out
