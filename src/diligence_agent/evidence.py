"""The unit every tool returns and every claim cites."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: str  # orange_book, ctgov, openfda_label, drugsfda, edgar, pubmed
    record_id: str  # stable within the source, e.g. N212273/patent/9012496
    url: str  # where a person can check the record
    text: str  # the passage a claim is checked against
    fields: dict[str, Any] = Field(default_factory=dict)  # structured values for code checks

    @property
    def key(self) -> str:
        return f"{self.source}:{self.record_id}"
