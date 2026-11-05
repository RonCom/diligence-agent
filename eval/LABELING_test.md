# Labeling guide: judge-graded answers

The judge failed calibration (reports/judge_calibration.json), so by the pre-registered rule these hand labels
replace its correctness grades on the judge-graded questions. Label `human_correct` in the CSV beside this guide;
`human_faithful` is not used here. Same rubric as eval/LABELING.md: 1 if the answer states the reference's key
facts without contradicting them, 0.5 if some but not all, 0 if it misses or contradicts them; a decline on an
answerable question is 0. An identical answer to the same question appears once and its label covers every repeat.

After labeling, run `uv run diligence-agent regrade --split <split>` for each system.

## The 12 answers

### L01 (tk02)

**Question:** What royalty rate does Royalty Pharma receive on Tremfya, according to its latest 10-K?

**Reference:** ~4% royalty (10-K Portfolio Summary).

**Answer:**

> The royalty rate for Tremfya is stated as approximately 4% [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
```

### L02 (tk03)

**Question:** What royalty rate does Royalty Pharma receive on Voranigo, according to its latest 10-K?

**Reference:** Tiered royalty of 15% on first $1 billion of U.S. sales, down to 12% on U.S. sales >$1 billion (10-K Portfolio Summary).

**Answer:**

> The royalty rate for Voranigo is a tiered royalty of 15% on the first $1 billion of U.S. sales, decreasing to 12% on U.S. sales exceeding $1 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
```

### L03 (tk04)

**Question:** What cut of Trodelvy sales does Royalty Pharma get?

**Reference:** Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion (10-K Portfolio Summary).

**Answer:**

> Royalty Pharma has a tiered royalty for Trodelvy: 4.15% on the first $2 billion in sales, which decreases to 1.75% on sales exceeding $6 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: Trodelvy | 2018 | Perpetual | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion | 82.4%
Erleada | 2019, 2023 | 2032 | Low-single digit royalty | 86.2%
Imdelltra (9) | 2025 | 2038-2041 | ~7% royalty with royalty sharing on sales >$1.5 billion | 100.0%
(1)Durations shown represent our estimates as of the current reporting date of when a royalty will substantially end, which may vary by geography and may depend on clinical trial results, regulatory approvals (including the timing of such approvals), contractual terms, commercial developments, estimates of regulatory exclusivity
```

### L04 (tk06)

**Question:** What royalty rate does Royalty Pharma receive on Imdelltra, according to its latest 10-K?

**Reference:** ~7% royalty with royalty sharing on sales >$1.5 billion (10-K Portfolio Summary).

**Answer:**

> Imdelltra has a tiered royalty structure: 4.15% on the first $2 billion, decreasing to 1.75% on sales exceeding $6 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2]. Additionally, there is a "~7% royalty with royalty sharing on sales over $1.5 billion" [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2]. Furthermore, the company is entitled to royalties on worldwide net sales of Imdelltra, excluding sales in China [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/4].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: Trodelvy | 2018 | Perpetual | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion | 82.4%
Erleada | 2019, 2023 | 2032 | Low-single digit royalty | 86.2%
Imdelltra (9) | 2025 | 2038-2041 | ~7% royalty with royalty sharing on sales >$1.5 billion | 100.0%
(1)Durations shown represent our estimates as of the current reporting date of when a royalty will substantially end, which may vary by geography and may depend on clinical trial results, regulatory approvals (including the timing of such approvals), contractual terms, commercial developments, estimates of regulatory exclusivity
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/4] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: (7)We are entitled to royalties on U.S. sales of cabozantinib products through September 2026 and non-U.S. markets through the full term of the royalty.
(8)Our royalty interest in Spinraza will revert to Ionis after we receive aggregate Spinraza royalties equal to $475 million or $550 million, depending on the timing and occurrence of certain events. We are entitled to 25% of Ionis’ Spinraza royalty payments of 11% to 15% on sales up to $1.5 billion through 2027, increasing to 45% of royalty payments on sales up to $1.5 billion in 2028.
(9)We are entitled to royalties on worldwide net sales of Imdelltra,
```

### L05 (tk02)

**Question:** What royalty rate does Royalty Pharma receive on Tremfya, according to its latest 10-K?

**Reference:** ~4% royalty (10-K Portfolio Summary).

**Answer:**

> The royalty rate for Tremfya is approximately 4% [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
```

### L06 (tk03)

**Question:** What royalty rate does Royalty Pharma receive on Voranigo, according to its latest 10-K?

**Reference:** Tiered royalty of 15% on first $1 billion of U.S. sales, down to 12% on U.S. sales >$1 billion (10-K Portfolio Summary).

**Answer:**

> Voranigo has a tiered royalty structure: 15% on the first $1 billion of U.S. sales, decreasing to 12% on U.S. sales exceeding $1 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1]. Additionally, Voranigo sales are not disclosed by Servier [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] (not among the records collected)
```

### L07 (tk04)

**Question:** What cut of Trodelvy sales does Royalty Pharma get?

**Reference:** Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion (10-K Portfolio Summary).

**Answer:**

> Trodelvy has a tiered royalty structure: 4.15% on the first $2 billion in sales, which decreases to 1.75% on sales exceeding $6 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2]. This is also listed as a tiered royalty of 4.15% on the first $2 billion, down to 1.75% on sales >$6 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: Trodelvy | 2018 | Perpetual | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion | 82.4%
Erleada | 2019, 2023 | 2032 | Low-single digit royalty | 86.2%
Imdelltra (9) | 2025 | 2038-2041 | ~7% royalty with royalty sharing on sales >$1.5 billion | 100.0%
(1)Durations shown represent our estimates as of the current reporting date of when a royalty will substantially end, which may vary by geography and may depend on clinical trial results, regulatory approvals (including the timing of such approvals), contractual terms, commercial developments, estimates of regulatory exclusivity
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
```

### L08 (tk06)

**Question:** What royalty rate does Royalty Pharma receive on Imdelltra, according to its latest 10-K?

**Reference:** ~7% royalty with royalty sharing on sales >$1.5 billion (10-K Portfolio Summary).

**Answer:**

> Royalty Pharma receives approximately a 7% royalty on Imdelltra, with royalty sharing on sales exceeding $1.5 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2]. Another source states that Imdelltra has a royalty of approximately 7% with royalty sharing on sales over $1.5 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: Trodelvy | 2018 | Perpetual | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion | 82.4%
Erleada | 2019, 2023 | 2032 | Low-single digit royalty | 86.2%
Imdelltra (9) | 2025 | 2038-2041 | ~7% royalty with royalty sharing on sales >$1.5 billion | 100.0%
(1)Durations shown represent our estimates as of the current reporting date of when a royalty will substantially end, which may vary by geography and may depend on clinical trial results, regulatory approvals (including the timing of such approvals), contractual terms, commercial developments, estimates of regulatory exclusivity
```

### L09 (tk02)

**Question:** What royalty rate does Royalty Pharma receive on Tremfya, according to its latest 10-K?

**Reference:** ~4% royalty (10-K Portfolio Summary).

**Answer:**

> The latest 10-K filing indicates that the royalty rate for Tremfya is approximately 4% [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
```

### L10 (tk03)

**Question:** What royalty rate does Royalty Pharma receive on Voranigo, according to its latest 10-K?

**Reference:** Tiered royalty of 15% on first $1 billion of U.S. sales, down to 12% on U.S. sales >$1 billion (10-K Portfolio Summary).

**Answer:**

> The royalty rate for Voranigo is a tiered royalty: 15% on the first $1 billion of U.S. sales, which then drops to 12% on U.S. sales exceeding $1 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1]. The estimated royalty duration for Voranigo is 2038 [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/1] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: The table below provides a summary of the acquisition year, estimated royalty duration, royalty rates and the ownership percentages attributable to Royalty Pharma, net of legacy non-controlling interests for selected approved products in our portfolio:
Products | Acquisition Year(s) | Estimated Royalty Duration (1) | Royalty Rates (2) | Attributable to Royalty Pharma (3)
Cystic fibrosis franchise (4) | 2014, 2020 | 2039-2041 | Blended royalty of slightly over 9% for Trikafta; See footnote (4) | 86.8%
Trelegy (5) | 2022 | 2029-2030 | Tiered royalty of 6.5% on first $750 million, up to 10% on sales >$2.25 
```

### L11 (tk04)

**Question:** What cut of Trodelvy sales does Royalty Pharma get?

**Reference:** Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion (10-K Portfolio Summary).

**Answer:**

> The royalty terms for Trodelvy are a tiered structure: 4.15% on the first $2 billion in sales, which decreases to 1.75% on sales exceeding $6 billion [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2]. Royalty Pharma holds an 82.4% ownership percentage for this royalty [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: Trodelvy | 2018 | Perpetual | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion | 82.4%
Erleada | 2019, 2023 | 2032 | Low-single digit royalty | 86.2%
Imdelltra (9) | 2025 | 2038-2041 | ~7% royalty with royalty sharing on sales >$1.5 billion | 100.0%
(1)Durations shown represent our estimates as of the current reporting date of when a royalty will substantially end, which may vary by geography and may depend on clinical trial results, regulatory approvals (including the timing of such approvals), contractual terms, commercial developments, estimates of regulatory exclusivity
```

### L12 (tk06)

**Question:** What royalty rate does Royalty Pharma receive on Imdelltra, according to its latest 10-K?

**Reference:** ~7% royalty with royalty sharing on sales >$1.5 billion (10-K Portfolio Summary).

**Answer:**

> The latest 10-K indicates that Royalty Pharma has a royalty interest in Imdelltra with a "~7% royalty with royalty sharing on sales >$1.5 billion" [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2]. Additionally, the company states they are entitled to royalties on worldwide net sales of Imdelltra, excluding sales in China [sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/4].

**Cited records:**

```
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/2] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: Trodelvy | 2018 | Perpetual | Tiered royalty of 4.15% on first $2 billion, down to 1.75% on sales >$6 billion | 82.4%
Erleada | 2019, 2023 | 2032 | Low-single digit royalty | 86.2%
Imdelltra (9) | 2025 | 2038-2041 | ~7% royalty with royalty sharing on sales >$1.5 billion | 100.0%
(1)Durations shown represent our estimates as of the current reporting date of when a royalty will substantially end, which may vary by geography and may depend on clinical trial results, regulatory approvals (including the timing of such approvals), contractual terms, commercial developments, estimates of regulatory exclusivity
[sec_filing:rprx-10-k-2025/item-1-business-portfolio-summary/4] Royalty Pharma plc 10-K, period ending 2025-12-31, Item 1. BUSINESS > Portfolio Summary: (7)We are entitled to royalties on U.S. sales of cabozantinib products through September 2026 and non-U.S. markets through the full term of the royalty.
(8)Our royalty interest in Spinraza will revert to Ionis after we receive aggregate Spinraza royalties equal to $475 million or $550 million, depending on the timing and occurrence of certain events. We are entitled to 25% of Ionis’ Spinraza royalty payments of 11% to 15% on sales up to $1.5 billion through 2027, increasing to 45% of royalty payments on sales up to $1.5 billion in 2028.
(9)We are entitled to royalties on worldwide net sales of Imdelltra,
```
