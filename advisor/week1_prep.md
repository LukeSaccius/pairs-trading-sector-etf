# Week 1 Advisor Prep

## What I did this week
- Automated ETF ingestion via `download_etf_data`, validation, and raw CSV persistence.
- Built and executed the pair-scanning pipeline (correlation + Engle–Granger + Johansen follow-up).
- Expanded the sector universe/metadata for both same-sector and cross-sector tests.
- Documented theory takeaways (see `notes/week1_concepts.md`).

## Preliminary findings
- Record counts of qualifying pairs split by `pair_bucket` (Same vs Cross Sector).
- Highlight Engle–Granger pass rate and any Johansen-confirmed trios.
- Note sectors with recurring cointegration hits (e.g., Tech vs Financials).

## Questions for my advisor
- Are p-value < 0.05 and corr > 0.85 strict enough for Week 1, or should thresholds vary by sector?
- Should we prioritize same-sector trades for now, or explore cross-sector spreads despite higher risk?
- How should we treat half-life estimates when they fluctuate widely across rolling windows?

## Next week plan
- Move into OU modeling / half-life calibration on shortlisted pairs.
- Prototype signal generation + basic backtests for validated pairs.
- Begin drafting Week 2 notebook structure (OU estimation + entry/exit logic).
