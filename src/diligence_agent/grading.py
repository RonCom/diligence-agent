"""Grade answers: code checks where the eval set decides exactly, an LLM judge for wording, and policy-rag's rules.

Code checks (score 1, 0.5 or 0):
  date          the gold date anywhere in the answer: 1 if it is the first date mentioned, 0.5 if a later one
  yes_no        the first sentence says yes / is approved, or no / is not approved
  trial_status  status and primary completion month both stated: 1; one of them: 0.5
  nct_set       F1 between the NCT ids in the answer and the gold set: 1 at F1 = 1, 0.5 at F1 >= 0.5
  money         a dollar amount within 1% of the gold value in millions ("$250 million", "$6.3 billion", "6,297")
  contains_all  every token appears in the answer (years, months)
  none_listed   says no Orange Book patents are listed (or that it's a biologic), or declines
  decline       declines (out-of-scope questions)
  judge         the judge grades against the reference and key facts

Rules from policy-rag (`code_rules`), applied after the check or the judge:
  - declines are detected in code: the exact reply, or a decline written as prose in the first sentence
  - a decline on an answerable question scores 0, exact or prose (biologic questions excepted: declining is right)
  - an exact decline is faithful
  - the judge's raw grades are kept beside the strict ones
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

import httpx

DECLINE = "NOT IN SOURCES"

# A decline written as prose instead of the exact reply, matched in the first sentence only, so an answer that notes
# a missing detail after answering still counts as an answer. Adapted from policy-rag's answer.declined.
_SRC = (
    r"(?:the\s+)?(?:provided\s+|given\s+|available\s+|returned\s+|retrieved\s+)?(?:sec\s+|fda\s+)?"
    r"(?:records?|sources?|tool\s+results?|search\s+results?|results?|data|documents?|information|tools?|"
    r"filings?|labels?)"
)
_PROSE = re.compile("|".join([
    rf"\b{_SRC}(?:\s+[\w.&'-]+){{0,6}}?\s+(?:do|does|did)\s+not\s+(?:provide|contain|include|address|specify|mention|state|cover|say|"
    rf"answer|show|list|report)",
    rf"\bnot\s+(?:addressed|covered|specified|mentioned|found|stated|included|provided|contained|available|"
    rf"listed|reported)\s+in\s+{_SRC}",
    r"\b(?:there\s+(?:is|are)\s+)?no\s+(?:information|records?|data|mention|results?)\s+"
    r"(?:in|within|about|regarding|on|for|available)\b",
    r"\b(?:cannot|can't|could\s+not|couldn't|unable\s+to)\s+(?:be\s+)?(?:answer|determine|find|tell|locate|"
    r"provide)\w*",
    r"\bI\s+(?:do\s+not|don't)\s+(?:know|have\s+(?:enough|specific|access|the\s+information))",
    r"\bnone\s+of\s+the\s+(?:listed\s+|returned\s+)?(?:trials|records|results|sources)",
    # Added after the test run (a documented deviation, config.yaml `deviations`): a search that came back empty,
    # as in LangChain's test answer to oos03, "...did not return any records."
    r"\b(?:did\s+not|didn't|does\s+not|doesn't)\s+return\s+any\s+(?:records?|results?|matches)",
    r"\breturned\s+no\s+(?:records?|results?|matches)",
]), re.IGNORECASE)  # fmt: skip


def declined(answer: str) -> str:
    """'exact' for the decline reply, 'prose' for a decline written out in the first sentence, '' otherwise."""
    text = unescape(answer).strip()
    if text.upper().startswith(DECLINE):
        return "exact"
    first = re.split(r"(?<=[.!?])\s+|\n", text, maxsplit=1)[0]
    return "prose" if _PROSE.search(first) else ""


# ---- parsing --------------------------------------------------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}  # fmt: skip
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(rf"\b{_MON}\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I), lambda m: (int(m[3]), _month(m[1]), int(m[2]))),
    (re.compile(rf"\b(\d{{1,2}})\s+{_MON},?\s+(\d{{4}})\b", re.I), lambda m: (int(m[3]), _month(m[2]), int(m[1]))),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), lambda m: (int(m[3]), int(m[1]), int(m[2]))),
]


def _month(name: str) -> int:
    name = name.lower().rstrip(".")
    return next(i for m, i in MONTHS.items() if m.startswith(name[:3]))


def dates_in(text: str) -> list[str]:
    """ISO dates in order of appearance."""
    found: list[tuple[int, str]] = []
    for pattern, parts in DATE_PATTERNS:
        for m in pattern.finditer(text):
            try:
                found.append((m.start(), date(*parts(m)).isoformat()))
            except (ValueError, StopIteration):
                continue
    return [d for _, d in sorted(found)]


def months_in(text: str) -> set[str]:
    """YYYY-MM for every full date and every 'Month YYYY' or 'YYYY-MM' in the text."""
    out = {d[:7] for d in dates_in(text)}
    out |= {f"{m[1]}-{m[2]}" for m in re.finditer(r"\b(\d{4})-(\d{2})\b(?!-\d)", text)}
    for m in re.finditer(rf"\b{_MON}\s+(\d{{4}})\b", text, re.I):
        out.add(f"{m[2]}-{_month(m[1]):02d}")
    return out


_MONEY = re.compile(r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(billion|bn|million|mn|thousand|m)?\b", re.I)
_SCALE = {"billion": 1000.0, "bn": 1000.0, "thousand": 0.001}  # to millions; 10-K tables are in thousands


def amounts_millions(text: str) -> list[float]:
    out = []
    for m in _MONEY.finditer(text):
        v = float(m[1].replace(",", ""))
        out.append(v * _SCALE.get((m[2] or "").lower(), 1.0))
    return out


# ---- checks ---------------------------------------------------------------------------------------------------------


def unescape(text: str) -> str:
    r"""Drop Markdown backslash escapes (ACTIVE\_NOT\_RECRUITING, \$170): formatting, not content."""
    return re.sub(r"\\([_*$#\\-])", r"\1", str(text))


def check(spec: dict, answer: str) -> float | None:
    """Score 0, 0.5 or 1 from the question's check spec; None when the judge grades it."""
    kind = spec["type"]
    text = unescape(answer)
    low = text.lower()
    if kind == "judge":
        return None
    if kind == "decline":
        return 1.0 if declined(text) else 0.0
    if kind == "date":
        found = dates_in(text)
        if spec["value"] not in found:
            return 0.0
        return 1.0 if found[0] == spec["value"] else 0.5
    if kind == "yes_no":
        if declined(text):  # "NOT IN SOURCES" leads with "not" but says neither yes nor no
            return 0.0
        first = re.split(r"(?<=[.!?])\s+|\n", text.strip(), maxsplit=1)[0].lower()
        lead = re.match(r"\W*(\w+)", first)
        word = lead[1] if lead else ""
        neg = re.search(r"\b(no|not|isn't|is not)\b", first) is not None
        pos = re.search(r"\b(yes|is (?:fda[- ])?approved|is indicated|approved for)\b", first) is not None
        said = True if word == "yes" else False if word == "no" else False if neg else True if pos else None
        return 1.0 if said is spec["value"] else 0.0
    if kind == "trial_status":
        status_ok = spec["status"].lower().replace("_", " ") in low.replace("_", " ").replace(",", "")
        month_ok = (spec["primary_completion"] or "")[:7] in months_in(text)
        return (status_ok + month_ok) / 2
    if kind == "nct_set":
        got = set(re.findall(r"NCT\d{8}", text.upper()))
        gold = set(spec["value"])
        if not gold:
            return 1.0 if not got else 0.0
        tp = len(got & gold)
        f1 = 2 * tp / (len(got) + len(gold)) if got else 0.0
        return 1.0 if f1 == 1 else 0.5 if f1 >= 0.5 else 0.0
    if kind == "money":
        gold = float(spec["value"])
        return 1.0 if any(abs(v - gold) <= 0.01 * gold for v in amounts_millions(text)) else 0.0
    if kind == "contains_all":
        return 1.0 if all(str(t).lower() in low for t in spec["value"]) else 0.0
    if kind == "none_listed":
        if declined(text):
            return 1.0
        pattern = (r"\b(no|not)\b.{0,40}\b(patents?|listed|listings?|records?|results?|entries)\b"
                   r"|\bbiologic|\bBLA\b|purple book")  # fmt: skip
        return 1.0 if re.search(pattern, text, re.I) else 0.0
    raise ValueError(f"unknown check {kind}")


# ---- judge ----------------------------------------------------------------------------------------------------------

JUDGE_SYSTEM = """You grade answers to drug diligence questions. Be strict and literal.
correct: 1 if the answer states the reference's key facts without contradicting them, 0.5 if it states some but
not all, 0 if it misses or contradicts them. If the reference is NOT IN SOURCES, correct is 1 only when the answer
declines and 0 if it gives a substantive answer.
faithful: 1 if every claim in the answer is supported by the records shown, 0 if any claim is not supported.
For a declined answer, faithful is 1."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "number", "enum": [0, 0.5, 1]},
        "faithful": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string"},
    },
    "required": ["correct", "faithful", "reason"],
}


def judge_prompt(q: dict, answer: str, records: str) -> str:
    return (f"Question: {q['question']}\nReference answer: {q['reference']}\n"
            f"Key facts: {'; '.join(q.get('key_facts') or []) or '-'}\n\n"
            f"Records shown to the answerer:\n{records or '(none)'}\n\nAnswer to grade:\n{answer}")  # fmt: skip


class OllamaJudge:
    def __init__(self, url: str, model: str, timeout_s: float = 900):
        self.url, self.model = url.rstrip("/"), model
        self.client = httpx.Client(timeout=timeout_s)

    def __call__(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": JUDGE_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 16384},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        r = self.client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]


def fake_judge(system: str, user: str) -> str:
    return json.dumps({"correct": 1, "faithful": 1, "reason": "fake judge"})


def judge(chat, q: dict, answer: str, records: str) -> dict[str, Any]:
    raw = re.sub(r"<think>.*?</think>", "", chat(JUDGE_SYSTEM, judge_prompt(q, answer, records)), flags=re.S).strip()
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        j = json.loads(m.group(0)) if m else {}
    return {"judge_correct": j.get("correct"), "judge_faithful": j.get("faithful"),
            "judge_reason": j.get("reason", raw[:200])}  # fmt: skip


# ---- per-answer grade -------------------------------------------------------------------------------------------


def grade(q: dict, answer: str, judged: dict | None = None, human: dict | None = None) -> dict[str, Any]:
    """Raw and strict grades for one answer. `judged` holds the judge's output for judge-checked questions.

    `human` is set when the judge failed calibration: per the pre-registered fallback, a hand label replaces the
    judge's correctness grade on judge-checked questions ({"correct": None} means the answer still needs a label).
    """
    kind = declined(answer)
    raw = check(q["check"], answer)
    judged = judged or {}
    source = "code"
    if raw is None:
        source = "human" if human is not None else "judge"
        raw = human.get("correct") if human is not None else judged.get("judge_correct")
    strict = raw
    decline_ok = q["check"]["type"] in ("decline", "none_listed")
    if kind and q["answerable"] and not decline_ok:
        strict = 0.0  # a decline on an answerable question is wrong, exact or prose
    faithful_raw = judged.get("judge_faithful")
    return {
        "abstain_kind": kind,
        "declined": bool(kind),
        "correct_raw": raw,
        "correct_strict": strict,
        "correct_source": source,
        "judge_correct": judged.get("judge_correct"),
        "judge_faithful_raw": faithful_raw,
        "judge_faithful": 1.0 if kind == "exact" else faithful_raw,  # an exact decline is faithful
        "judge_reason": judged.get("judge_reason", ""),
        "check_type": q["check"]["type"],
        "graded_at": datetime.now().isoformat(timespec="seconds"),
    }
