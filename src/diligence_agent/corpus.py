"""Search corpus: SEC filings split by their own Item and subsection headings, and FDA label sections.

10-K HTML (inline XBRL) puts every phrase in its own <span>, and its tables split "$" and footnote markers into
separate cells. Block elements become line breaks and spans are joined, so sentences stay whole; each table row
becomes one line of cells joined by " | ", so a row such as "Xtandi | 2016 | 2027-2028 | ..." stays together in one
chunk. Chunks break on line boundaries where they can, so a table row is never split.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

BLOCKS = ["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"]
ITEM = re.compile(r"^item\s+(\d+[a-c]?)\.?\s+(\S.*)$", re.I)
PART = re.compile(r"^part\s+[ivx]+\b", re.I)
NOISE = re.compile(r"^(\d{1,3}|table of contents|[ivx]+)$", re.I)
MARKER = re.compile(r"^\(\d+\)$")  # table footnote markers: (1), (2)

LABEL_SECTIONS = [
    "boxed_warning", "indications_and_usage", "dosage_and_administration", "dosage_forms_and_strengths",
    "contraindications", "warnings_and_cautions", "adverse_reactions", "use_in_specific_populations",
    "clinical_studies", "description", "mechanism_of_action",
]  # fmt: skip


@dataclass
class Chunk:
    chunk_id: str  # citation key within the source, e.g. rprx-10k-2025/item-1-portfolio-summary/2
    source: str  # sec_filing or label_text
    doc: str  # human label, e.g. "Royalty Pharma plc 10-K, period 2025-12-31"
    section: str  # e.g. "Item 1. Business > Portfolio Summary"
    url: str
    text: str
    drug: str | None = None  # labels: the product; filings: None (filter by mention instead)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def search_text(self) -> str:
        return f"{self.section}. {self.text}"


def _cell(td) -> str:
    return " ".join(td.get_text(" ").split())


def table_rows(table) -> list[str]:
    rows = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for c in (_cell(td) for td in tr.find_all(["td", "th"])):
            if not c or c in {"$", "%", ")", "—"} and not cells:
                continue
            if c in {"$", "%", ")"} or MARKER.match(c):
                if cells and MARKER.match(c):
                    cells[-1] += f" {c}"
                continue
            cells.append(c)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def html_lines(html: bytes | str) -> list[str]:
    if isinstance(html, bytes):  # decode here: filings declare encoding='ASCII' and the parser then mangles entities
        try:
            html = html.decode("utf-8")
        except UnicodeDecodeError:
            html = html.decode("cp1252")
    html = re.sub(r"^<\?xml[^>]*\?>", "", html.lstrip())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    for t in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
        t.decompose()  # the hidden ix:header block of inline XBRL
    for table in soup.find_all("table"):
        table.replace_with(NavigableString("\n" + "\n".join(table_rows(table)) + "\n"))
    for tag in soup.find_all(BLOCKS):
        tag.insert_after("\n")
    lines = (" ".join(line.split()) for line in soup.get_text("").splitlines())
    return [line for line in lines if line and not NOISE.match(line)]


def _is_subheading(line: str, nxt: str | None) -> bool:
    words = line.split()
    return (
        1 <= len(words) <= 10
        and line[0].isupper()
        and not line.endswith((".", ":", ";", ","))
        and " | " not in line
        and not line.startswith(("•", "(", "Item ", "ITEM "))
        and nxt is not None
        and (len(nxt.split()) >= 15 or " | " in nxt)
    )


def sectioned(lines: list[str]) -> list[tuple[str, list[str]]]:
    """[(section label, lines)], starting at the first Item heading after the table of contents."""
    out: list[tuple[str, list[str]]] = []
    item, sub, body = None, None, []

    def flush():
        if item and body:
            out.append((f"{item} > {sub}" if sub else item, body[:]))
        body.clear()

    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        m = ITEM.match(line)
        if m and " | " not in line and len(line) < 200:
            flush()
            item, sub = f"Item {m.group(1).upper()}. {m.group(2).rstrip('.')}", None
            continue
        if PART.match(line) and len(line) < 60:
            continue
        if item and _is_subheading(line, nxt):
            flush()
            sub = line
            continue
        if item:
            body.append(line)
    flush()
    return out


def _windows(lines: list[str], max_words: int, overlap: int) -> list[str]:
    """Pack whole lines into chunks of about max_words, carrying about `overlap` words of trailing lines forward."""
    pieces: list[str] = []
    for line in lines:  # split only lines that alone exceed the budget
        w = line.split()
        if len(w) <= max_words:
            pieces.append(line)
        else:
            step = max_words - overlap
            pieces.extend(" ".join(w[i : i + max_words]) for i in range(0, len(w), step) if w[i : i + max_words])
    chunks, cur, n = [], [], 0
    for p in pieces:
        k = len(p.split())
        if cur and n + k > max_words:
            chunks.append("\n".join(cur))
            carry, c = [], 0
            for q in reversed(cur):
                if c + len(q.split()) > overlap:
                    break
                carry.insert(0, q)
                c += len(q.split())
            cur, n = carry, c
        cur.append(p)
        n += k
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:90].strip("-")


def filing_chunks(html: bytes | str, doc_key: str, doc: str, url: str, max_words=300, overlap=40) -> list[Chunk]:
    out = []
    seen: dict[str, int] = {}
    for section, body in sectioned(html_lines(html)):
        base = slug(section)  # citation keys must be unique: a repeated heading gets -2, -3
        seen[base] = seen.get(base, 0) + 1
        key = base if seen[base] == 1 else f"{base}-{seen[base]}"
        for n, text in enumerate(_windows(body, max_words, overlap), 1):
            out.append(Chunk(f"{doc_key}/{key}/{n}", "sec_filing", doc, section, url, text))
    return out


def label_chunks(label: dict, drug: str, url: str, max_words=300, overlap=40) -> list[Chunk]:
    effective = label.get("effective_time", "")
    doc = f"{drug} FDA label, effective {effective[:4]}-{effective[4:6]}-{effective[6:]}"
    out = []
    for section in LABEL_SECTIONS:
        body = " ".join(label.get(section, []))
        if not body:
            continue
        for n, text in enumerate(_windows([body], max_words, overlap), 1):
            key = f"{label['set_id']}@{effective}/{section}/{n}"
            title = section.replace("_", " ").capitalize()
            out.append(Chunk(key, "label_text", doc, title, url, text, drug=drug))
    return out
