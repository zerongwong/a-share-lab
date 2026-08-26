---
name: a-share-midterm-research
description: Build or review an evidence-based four-stock A-share research portfolio for holding periods from one week to one year, with one to three months as the default. Use when the user asks for A-share screening, a medium-term portfolio update, entry or invalidation levels, or a review combining CSMAR history, current LongBridge evidence, fundamentals, market regime, official announcements, and news.
---

# A-share medium-term research

Read [references/model-contract.md](references/model-contract.md) before producing a portfolio or changing the model.

## Workflow

1. Resolve the latest common data cutoff. In live mode, refuse to present a stale stock cutoff as current. Never combine a newer index observation with an older stock universe without marking and correcting the mismatch.
2. Run the deterministic full-market screen first. Do not let an LLM invent prices, indicators, scores, probabilities, Sharpe ratios, or drawdowns.
3. Apply all hard eligibility, data-quality, execution, and late-stage acceleration gates. Returning no portfolio is valid.
4. Use current fundamentals, LongBridge read-only data, official announcements, and web news only after the quantitative screen has reduced the universe. Treat current snapshots as current evidence, not historical point-in-time data. A balance-sheet snapshot may enter only after both the decision date and price cutoff pass its conservative retrieval-time gate.
5. Review the finalists with explicit bull, bear, data-quality, and portfolio-risk checks. Resolve conflicts from primary evidence rather than by majority vote.
6. Produce at most one four-stock portfolio. Use total-account weights 30%, 20%, 20%, and 10%, retain 20% cash, use no financing, and never place orders.
7. Archive the cutoff, input hashes, strategy version, exclusions, evidence links, factor coverage, and output before presenting the result.

## Tool boundaries

- Use CSMAR or another licensed point-in-time source for historical prices and backtests. Never upload or redistribute licensed raw data.
- Use LongBridge only through read-only quote, static-info, market, financial, valuation, candlestick, and news capabilities. Never use account, position, watchlist mutation, alert mutation, or trading capabilities.
- Prefer exchange, CNINFO, issuer, regulator, and government sources for material facts. Preserve title, publisher, publication time, retrieval time, and URL.
- Treat open-web search as finalist due diligence, not a complete or reproducible full-market dataset.
- Keep all credentials in the user's own connector or operating-system credential store. Never write them to the repository, report, database, prompt, or logs.

## Output rules

- State the data cutoff and data-quality limitations first.
- Distinguish one-week entry timing, one-to-three-month core thesis, six-month validation, and one-year continuation conditions.
- Show every hard-gate exclusion that materially changed the result, especially limit-up, suspension, unbuyable, late-stage acceleration, accounting quality, and concentration gates.
- Report historical or walk-forward metrics with their method and confidence interval. Do not label in-sample statistics as forecasts.
- Present scenario ranges, not guaranteed returns. If probability calibration is unavailable, say “not estimated.”
- Give entry zones, breakout confirmation, structural invalidation, and staged reduction zones as conditional plans, never instructions to buy immediately.
- End with the exact conditions that would keep the allocation in cash or trigger a new review.
