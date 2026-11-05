"""ClinicalTrials.gov API v2 study search."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from diligence_agent.evidence import Evidence
from diligence_agent.tools.base import Tool, ToolContext

API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,BriefTitle,OverallStatus,Phase,PrimaryCompletionDate,LeadSponsorName,Condition,InterventionName"

Phase = Literal["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
Status = Literal[
    "NOT_YET_RECRUITING",
    "RECRUITING",
    "ENROLLING_BY_INVITATION",
    "ACTIVE_NOT_RECRUITING",
    "SUSPENDED",
    "TERMINATED",
    "COMPLETED",
    "WITHDRAWN",
    "UNKNOWN",
]


class CtgovArgs(BaseModel):
    intervention: str = Field(min_length=2, description="Drug name, brand or generic")
    condition: str | None = Field(None, description="Disease or condition, optional")
    phase: Phase | None = Field(None, description="Trial phase, e.g. PHASE3")
    status: list[Status] | None = Field(None, description="Overall statuses to keep, e.g. [RECRUITING]")
    primary_completion_from: date | None = Field(None, description="Earliest primary completion date, YYYY-MM-DD")
    primary_completion_to: date | None = Field(None, description="Latest primary completion date, YYYY-MM-DD")
    max_results: int = Field(20, ge=1, le=50)

    # A small model writes "Phase 3" or "active, not recruiting"; map those to the API's enum values.
    @field_validator("phase", mode="before")
    @classmethod
    def _phase(cls, v):
        if isinstance(v, str | int):
            s = re.sub(r"[^A-Z0-9]", "", str(v).upper())
            s = s if s.startswith(("PHASE", "EARLYPHASE")) else f"PHASE{s}"
            return s.replace("EARLYPHASE", "EARLY_PHASE")
        return v

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [re.sub(r"[^A-Z]+", "_", str(s).upper()).strip("_") for s in v]
        return v


def study_evidence(study: dict) -> Evidence:
    ps = study.get("protocolSection", {})
    nct = ps.get("identificationModule", {}).get("nctId", "")
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    status = ps.get("statusModule", {})
    pcd = status.get("primaryCompletionDateStruct", {})
    phases = ps.get("designModule", {}).get("phases", [])
    sponsor = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")
    conditions = ps.get("conditionsModule", {}).get("conditions", [])
    interventions = [i.get("name", "") for i in ps.get("armsInterventionsModule", {}).get("interventions", [])]
    pcd_type = (pcd.get("type") or "").lower()
    return Evidence(
        source="ctgov",
        record_id=nct,
        url=f"https://clinicaltrials.gov/study/{nct}",
        text=(
            f"{nct}: {title}. Phase: {', '.join(phases) or 'not applicable'}. "
            f"Status: {status.get('overallStatus', 'unknown')}. Primary completion: {pcd.get('date', 'not given')}"
            f"{f' ({pcd_type})' if pcd_type else ''}. Lead sponsor: {sponsor}. Conditions: {', '.join(conditions)}. "
            f"Interventions: {', '.join(interventions)}."
        ),
        fields={
            "nct_id": nct,
            "phases": phases,
            "status": status.get("overallStatus"),
            "primary_completion": pcd.get("date"),
            "primary_completion_type": pcd.get("type"),
        },
    )


class CtgovTool(Tool):
    name = "ctgov_search"
    description = (
        "Search ClinicalTrials.gov for studies of a drug. Returns each study's NCT id, title, phase, overall status, "
        "primary completion date (actual or estimated) and lead sponsor. Filter by phase, status and a primary "
        "completion date range to find upcoming readouts."
    )
    Args = CtgovArgs

    def run(self, args: CtgovArgs, ctx: ToolContext) -> list[Evidence]:
        params = {"query.intr": args.intervention, "pageSize": args.max_results, "fields": FIELDS}
        if args.condition:
            params["query.cond"] = args.condition
        if args.status:
            params["filter.overallStatus"] = ",".join(args.status)
        advanced = []
        if args.phase:
            advanced.append(f"AREA[Phase]{args.phase}")
        if args.primary_completion_from or args.primary_completion_to:
            lo = args.primary_completion_from.isoformat() if args.primary_completion_from else "MIN"
            hi = args.primary_completion_to.isoformat() if args.primary_completion_to else "MAX"
            advanced.append(f"AREA[PrimaryCompletionDate]RANGE[{lo},{hi}]")
        if advanced:
            params["filter.advanced"] = " AND ".join(advanced)
        resp = ctx.http.get(API, params=params)
        if resp.status_code == 404:
            return []
        return [study_evidence(s) for s in resp.json().get("studies", [])]
