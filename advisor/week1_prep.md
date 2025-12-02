# Week 1 Advisor Prep

**Last Updated:** 2025-12-03

## What I did this week

### Phase 1: Initial Setup
- Automated ETF ingestion via `download_etf_data`, validation, and raw CSV persistence.
- Built and executed the pair-scanning pipeline (correlation + Engle–Granger + Johansen follow-up).
- Expanded the sector universe/metadata for both same-sector and cross-sector tests.
- Documented theory takeaways (see `notes/week1_concepts.md`).

### Phase 2: Tokat Methodology Replication
- Discovered Tokat & Hayrullahoglu (2021) paper claiming 15% annual return on ETF pairs
- Implemented full walk-forward backtest (252d formation → 252d trading → annual rebalance)
- Fixed critical bugs: PnL calculation, exit conditions, position sizing
- Extended data download: 2006-2025 (5,010 days, 134 ETFs)

### Phase 3: Fresh Data Verification (Latest)
- Complete data reset and fresh download
- Ran all tests (11/11 passed)
- Verified Engle-Granger on real ETF data (SPY-IVV, SPY-VOO, GLD-IAU)
- Performed rolling consistency analysis with 252d/252d parameters

## Key Findings

### Finding 1: ETFs ARE Cointegrated But NOT Tradeable
| Pair | Cointegration p-value | Half-Life | Tradeable? |
|------|----------------------|-----------|------------|
| GLD-IAU | 0.0001 ✅ | 628,182 days | ❌ (1,721 years) |
| SPY-VOO | 0.0002 ✅ | 89,657 days | ❌ (246 years) |

### Finding 2: Zero Tradeable Pairs with Standard Filters
```
Pairs passing p-value filter (p < 0.10): ~100+
Pairs passing half-life filter (15-120d): 16
Pairs passing 30%+ rolling consistency: 0
Best pair consistency: 14.5% (SPY-IVV)
```

### Finding 3: Strategy Works ONLY in Crisis
| Period | Avg Return | Win Rate | Interpretation |
|--------|------------|----------|----------------|
| Crisis (2008-2009) | +2.3% | 79.8% | ✅ Works |
| Non-Crisis (2011-2024) | -0.4% | 57.3% | ❌ Fails |

### Finding 4: Gap vs Tokat Paper Explained
| Factor | Impact |
|--------|--------|
| Universe (84% stocks in paper vs 100% ETFs) | -25 to -30% |
| Time period (paper includes 2008 crisis) | -5 to -10% |
| Possible data snooping in paper | Unknown |

## Questions for my advisor

1. **Should we pivot?**
   - ETF-only pairs trading appears non-viable
   - Options: individual stocks, factor-based strategies, or document negative finding

2. **Is this a valid research contribution?**
   - Replicating Tokat paper with negative results is scientifically valuable
   - Demonstrates ETF market efficiency and alpha decay

3. **Half-life problem:**
   - Literature rarely discusses the gap between "cointegrated" and "tradeable"
   - Is this worth writing up as a methodological note?

4. **What's the minimum bar for success?**
   - If we can't beat 0% return, should we conclude the strategy is dead?
   - Or try increasingly creative approaches?

## Updated Next Steps

### Option A: Document Negative Finding (Recommended)
- Write up thorough analysis of why ETF pairs trading doesn't work
- Emphasize cointegration ≠ tradeability lesson
- Valuable contribution to literature

### Option B: Pivot to Stocks
- Expand universe to include individual stocks (as Tokat paper does)
- Higher transaction costs but potentially more opportunities
- Requires new data download

### Option C: Regime-Aware Trading
- Only trade during high-volatility regimes
- Requires regime detection model
- Complex implementation

### Option D: Factor-Based Spreads
- Trade factor exposure differences rather than price cointegration
- More sophisticated approach
- May require additional factor data

## Files Created/Modified

- `data/raw/etf_prices_fresh.csv` - Fresh price data (2006-2025)
- `scripts/check_rolling_consistency.py` - Rolling consistency checker
- `scripts/check_pvalue_only.py` - P-value only checker
- `scripts/tokat_walkforward_backtest.py` - Full backtest implementation
- `docs/research_log.md` - Comprehensive documentation
