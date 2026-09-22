---
name: a-share-midterm-research
description: Build or review an evidence-based, low-frequency A-share research portfolio using entry and exit signals, locked existing holdings, and portfolio-aware replacements. Use for A-share screening, portfolio or protective-level reviews, and evidence-based strategy audits; retain fixed-horizon research only when explicitly requested.
---

# A-share medium-term research

Read [references/model-contract.md](references/model-contract.md) before producing a portfolio or changing the model.

## Workflow

1. Resolve the latest common data cutoff. In live mode, refuse to present a stale stock cutoff as current. Never combine a newer index observation with an older stock universe without marking and correcting the mismatch.
2. Run the deterministic full-market screen first. Do not let an LLM invent prices, indicators, scores, probabilities, Sharpe ratios, or drawdowns.
3. Require completed-weekly uptrend and confirmed weekly base breakout/healthy retest, then completed-daily evidence of a reasonable execution position in that structure. Apply all hard eligibility, data-quality, execution, and late-stage acceleration gates. A partial week or daily-only breakout cannot confirm weekly structure. Returning no portfolio is valid.
4. Only after the technical screen has reduced the universe, verify current financial quality and official announcements before portfolio optimisation. Use current fundamentals, LongBridge read-only data and primary public sources with publication and retrieval timestamps. Financial/numeric quality and official announcement listing/content review are separate evidence checks; a retrieved list is not a completed review or automatic pass. Confirm the latest applicable publicly disclosed report, distinguish financial from nonfinancial firms, and keep missing evidence unknown. Treat current snapshots as current evidence, not historical point-in-time data. A balance-sheet snapshot may enter only after both the decision date and price cutoff pass its conservative retrieval-time gate. Never claim stable future returns merely because a company passes a financial-quality screen.
5. Review the finalists with explicit bull, bear, data-quality, and portfolio-risk checks. Resolve conflicts from primary evidence rather than by majority vote.
6. Compare all qualified one-to-five-name initial sets with cash; zero means cash, and one/two are genuine low-exposure allocations rather than fallbacks after three-to-five-name failure. Rank by the frozen total-account historical return lower bound and joint risk; count preference is only a tie-break. Never weaken a gate to fill a count. Keep at most one name per industry, cap every new name at 20% of total account equity, and prefer roughly 15% when the evidence and exposure budget permit. The one-to-five-name exposure ceilings are 15%, 30%, 45%, 60%, 75%, lowered further by the cycle ceiling. For an existing portfolio lock confirmed membership and drifted weights; compare all eligible single replacements jointly with retained holdings and cash. Do not turn an exit recommendation into a fill or spend unconfirmed proceeds. Missing account or corporate-action evidence blocks replacement, not permission to invent a new portfolio. No financing or orders.
7. Archive the cutoff, input hashes, strategy version, exclusions, evidence links, factor coverage, and output before presenting the result.

## Tool boundaries

- Use CSMAR or another licensed point-in-time source for historical prices and backtests. Never upload or redistribute licensed raw data.
- Use LongBridge only through read-only quote, static-info, market, financial, valuation, candlestick, and news capabilities. Never use account, position, watchlist mutation, alert mutation, or trading capabilities.
- Prefer exchange, CNINFO, issuer, regulator, and government sources for material facts. Preserve title, publisher, publication time, retrieval time, and URL.
- Treat open-web search as finalist due diligence, not a complete or reproducible full-market dataset.
- Keep all credentials in the user's own connector or operating-system credential store. Never write them to the repository, report, database, prompt, or logs.

## Output rules

- State the data cutoff and data-quality limitations first.
- Default to one `continuous-signal-v4` plan without forced maturity, using completed-weekly direction plus confirmed weekly breakout/healthy retest, followed by completed-daily execution evidence for that same structure. A daily breakout alone cannot replace missing weekly confirmation. A base/range reversal may enter only after independent confirmation; early and orderly non-extended uptrends may also enter. Early location is a ranking preference, not an eligibility veto. Financial and official-announcement quality evidence is required before joint portfolio optimisation; numeric tests or a downloaded announcement list do not prove complete automated due diligence. Show urgent holding risks first, then concise conditional entries, total-account weights, protection lines and cash. `continuous-count-policy-v4.0.0` compares zero-to-five outcomes without mandatory three-to-five precedence. Show up to five ranked technical candidates separately from permission to enter, stating risk or missing-evidence reasons. Fixed-horizon legacy comparisons must be labelled read-only compatibility research and cannot masquerade as the live replacement plan.
- The single-stock 8% rule is a loss threshold against explicitly confirmed actual investment cost: cost × 92%, not 8% of the whole portfolio or an obligatory maximum structure-distance admission gate. Keep structural distance as risk information; retain non-extended execution and joint risk checks. An earlier confirmed structural failure can trigger an earlier exit review, and stored protection lines must never move down. Gaps, T+1, suspension, limit-down and costs mean trigger loss is not guaranteed realised loss; alerts are not fills.
- Show every hard-gate exclusion that materially changed the result, especially limit-up, suspension, unbuyable, late-stage acceleration, accounting quality, and concentration gates.
- Report historical or walk-forward metrics with their method and confidence interval. Do not label in-sample statistics as forecasts.
- Present scenario ranges, not guaranteed returns. If probability calibration is unavailable, say “not estimated.”
- Give entry zones, breakout confirmation, structural invalidation, and staged reduction zones as conditional plans, never instructions to buy immediately.
- End with the exact conditions that would keep the allocation in cash or trigger a new review.
