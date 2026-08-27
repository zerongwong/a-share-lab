# Medium-term model contract

## Objective

Select one diversified 3-to-5-stock research portfolio that maximizes the lower-confidence bound of net holding-period return subject to hard downside-volatility, rolling-drawdown, five-day expected-shortfall, down-correlation, risk-contribution, liquidity, execution, concentration, and data-quality constraints. Do not optimize raw return alone and do not ask the user to choose an annual return target.

Four holdings are the attention default. Three holdings use 70% stock exposure, four use 80%, and five use 85%; the remainder is cash. Weights are generated from inverse downside volatility with a bounded signal tilt and explicit position/industry caps. Financing is zero. The legacy fixed 30% / 20% / 20% / 10% allocation is compatibility-only.

## Horizons

| Horizon | Role |
|---|---|
| 1 week / 5 sessions | Entry timing and immediate failure detection |
| 1 month / 20 sessions | Short end of the core holding thesis |
| 3 months / 60 sessions | Default selection and portfolio-risk horizon |
| 6 months / 120 sessions | Fundamental and trend persistence check |
| 1 year / 252 sessions | Long-term continuation scenario, never extrapolated from short momentum |

Require the thesis to remain coherent across adjacent horizons. A strong five-day move cannot compensate for a late-stage or unstable 60/120-day structure.

## Data hierarchy

1. Licensed historical stock, corporate-action, membership, financial, and publication-time data.
2. Exchange/CNINFO announcements and issuer filings.
3. Licensed current quotes, market regime, valuations, and candidate news from LongBridge or an interchangeable provider.
4. Open-web finalist research with source and publication timestamps.

Never use a report period as its publication date. Never backfill today's revised financial statement, current industry classification, or current snapshot into an earlier decision.

## Hard gates

- Exclude ST/*ST, delisting, suspended, unbuyable, newly listed without sufficient history, and insufficiently liquid securities.
- Exclude formation-date limit-up names unless a later executable entry is independently confirmed.
- Freeze late acceleration when any robust combination holds: consecutive limit-ups, 20-day return above 35%, price more than 15% above MA20, price more than 3 ATR above its base, or a near-vertical Edwards–Magee trend channel. Re-evaluate only after a base forms.
- Reject unknown units, stale cutoffs, material missing eligibility fields, implausible corporate-action jumps, or unlicensed data.
- Reject portfolios that breach industry, pair-correlation, volatility, drawdown, liquidity-capacity, or data-coverage limits.

## Evidence layers

1. Market regime: index trend, breadth, volatility, drawdown, turnover, and cross-stock correlation.
2. Industry context: relative strength, breadth, valuation, and concentration. Use point-in-time industry membership for historical work.
3. Fundamentals: growth, profitability, cash conversion, balance-sheet strength, dilution, and industry-relative valuation. Treat financial and non-financial firms separately.
4. Trend and stage: Edwards–Magee primary/secondary trend, base, breakout, failed breakout, support/resistance, volume confirmation, and stage maturity.
5. Livermore execution discipline: act only at confirmed pivotal points, add only to profitable positions, avoid averaging down, and wait when evidence conflicts. Express fixed historical dollar rules with ATR/volatility-scaled thresholds.
6. Catalysts and risks: official filings first, then licensed or attributable news. Separate new information from already-priced narrative.
7. Portfolio construction: expected net return distribution, downside deviation, CVaR, maximum drawdown, Sharpe/Sortino/Calmar confidence bounds, correlation, and turnover costs.

Use a transparent deterministic rank ensemble before any language-model review. The language model may explain and challenge structured evidence but must not manufacture factor values.

The cross-stock breadth gate and the core-index gate are independent. If either is explicitly risk-off, pause new portfolio formation. If configured index evidence is stale or unavailable, fail closed rather than silently dropping the confirmation layer. A current-only balance-sheet rank is a narrow solvency/liquidity proxy, not the complete fundamentals layer, and is forbidden in historical replay.

## Validation before claiming a production result

- Use purged walk-forward evaluation with an embargo at least as long as the decision-to-execution gap.
- Form the portfolio at each historical decision date from the universe and information genuinely known then; do not select today's four stocks and backfill them through history.
- Include delisted names, historical ST/suspension states, corporate actions, fees, tax, slippage, and unfilled entries.
- Report block-bootstrap confidence intervals for Sharpe and Calmar, non-overlapping horizon outcomes, turnover, capacity, worst fold, maximum drawdown, and CVaR.
- Keep a research-only status until the out-of-sample thresholds are met. Never substitute in-sample metrics.

## Final report contract

Return either `DATA_NOT_READY`, `VALIDATION_NOT_READY`, `NO_ELIGIBLE_PORTFOLIO`, or `RESEARCH_ONLY`.

Only `RESEARCH_ONLY` may contain three to five securities. For each security include rank, automatic weight, thesis by horizon, entry condition, structural invalidation, reduction conditions, evidence, principal risks, and data confidence. For the portfolio include cash, zero borrowing, historical and out-of-sample metrics separately, scenario ranges, evidence coverage, strategy/data versions, and the next scheduled review condition.
