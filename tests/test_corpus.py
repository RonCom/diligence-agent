"""10-K parsing on a small synthetic filing that copies the real one's quirks."""

from __future__ import annotations

from diligence_agent.corpus import _windows, filing_chunks, html_lines, label_chunks, sectioned

LONG = " ".join(["word"] * 20)

FILING = f"""<?xml version='1.0' encoding='ASCII'?>
<html><head><title>x</title></head><body>
<div style="display:none"><ix:header>hidden xbrl facts</ix:header></div>
<table>
 <tr><td>Item 1.</td><td>Business</td><td>4</td></tr>
 <tr><td>Item 7.</td><td>Management&#8217;s Discussion</td><td>45</td></tr>
</table>
<div><span>PART I</span></div>
<div><span>Item 1. </span><span>BUSINESS</span></div>
<div><span>Portfolio Summary</span></div>
<div><span>The table below </span><span>summarizes royalties. {LONG}</span></div>
<table>
 <tr><td>Products</td><td>Royalty Rates</td><td>(2)</td></tr>
 <tr><td>Xtandi</td><td>$</td><td>Slightly less than 4% royalty</td><td>%</td><td>82.4</td></tr>
</table>
<div>17</div>
<div><span>Item 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS</span></div>
<div><span>As of December 31, 2024</span></div>
<div>Prior-year table follows. {LONG}</div>
<div><span>As of December 31, 2024</span></div>
<div>A second table under the same heading. {LONG}</div>
</body></html>""".encode()


def test_lines_join_spans_and_flatten_tables():
    lines = html_lines(FILING)
    assert "The table below summarizes royalties. " + LONG in lines
    assert "Xtandi | Slightly less than 4% royalty | 82.4" in lines  # "$" and "%" cells dropped
    assert "Products | Royalty Rates (2)" in lines  # footnote marker joins its cell
    assert not any("hidden xbrl" in line for line in lines)
    assert "17" not in lines  # page numbers dropped
    assert any("MANAGEMENT’S" in line for line in lines)  # entity decoded despite encoding='ASCII'


def test_sections_skip_table_of_contents():
    secs = dict(sectioned(html_lines(FILING)))
    assert "Item 1. BUSINESS > Portfolio Summary" in secs
    assert "Item 7. MANAGEMENT’S DISCUSSION AND ANALYSIS > As of December 31, 2024" in secs
    assert not any("Business" == s for s in secs)  # the TOC row "Item 1. | Business | 4" is not a heading


def test_chunk_keys_unique_for_repeated_headings():
    chunks = filing_chunks(FILING, "x-10-k-2025", "X 10-K", "https://sec.test/x.htm")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "x-10-k-2025/item-7-management-s-discussion-and-analysis-as-of-december-31-2024-2/1" in ids
    row = next(c for c in chunks if "Xtandi" in c.text)
    assert row.section == "Item 1. BUSINESS > Portfolio Summary"


def test_windows_keep_rows_whole_and_overlap():
    rows = [f"row {i} | " + " ".join(["v"] * 8) for i in range(30)]  # 11 words each
    chunks = _windows(rows, max_words=50, overlap=12)
    assert all(len(c.split()) <= 50 for c in chunks)
    for c in chunks:
        assert all(line.startswith("row ") and line.count("v") == 8 for line in c.splitlines())
    first, second = chunks[0].splitlines(), chunks[1].splitlines()
    assert second[0] == first[-1]  # one 11-word row of overlap


def test_label_chunks_by_section():
    label = {
        "set_id": "abc",
        "effective_time": "20260622",
        "indications_and_usage": ["1 INDICATIONS Treats CF."],
        "clinical_studies": [" ".join(["s"] * 700)],
    }
    chunks = label_chunks(label, "Trikafta", "https://dailymed.test", max_words=300, overlap=40)
    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "abc@20260622/indications_and_usage/1"
    assert sum("clinical_studies" in i for i in ids) == 3
    assert chunks[0].doc == "Trikafta FDA label, effective 2026-06-22" and chunks[0].drug == "Trikafta"
