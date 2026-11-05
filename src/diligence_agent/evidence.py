"""The unit every tool returns and every claim cites."""

from __future__ import annotations

import re
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


_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_KEY = re.compile(r"^[a-z_]+:\S+$")


def _keys(inside: str) -> list[str]:
    return [p for p in re.split(r"[,;\s]+", inside.strip()) if _KEY.match(p)]


def citations(text: str) -> list[str]:
    """Record keys cited in square brackets, one or several per bracket: [a:1] or [a:1, b:2]."""
    return [k for m in _BRACKET.finditer(str(text)) for k in _keys(m[1])]


def strip_citations(text: str) -> str:
    return " ".join(_BRACKET.sub(lambda m: "" if _keys(m[1]) else m[0], str(text)).split())
