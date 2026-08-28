# Medium-term model contract

## Objective

On every data-qualified decision date, run the deterministic full-market security screen even when the price-cycle evidence is defensive. When at least three securities pass the security gates, expose three to five ranked research candidates, with four as the attention default, and then separately determine which candidates, if any, have a conditional entry and whether they form a risk-feasible portfolio. If only zero to two securities pass the security gates, expose the available evidence, do not form a portfolio, and do not add inferior names. Do not optimize raw return alone and do not ask the user to choose an annual return target.

Candidate membership and deployment are separate decisions. A defensive cycle may tighten entry confirmation, lower the downside-risk budget, and reduce total stock exposure, but it must not suppress candidate discovery or be reported as a data failure. It is valid for a report to contain three to five research candidates and zero candidates currently labelled `CONDITIONAL_ENTRY`.

If no evaluated set passes every portfolio-risk and holding-period-return gate, the report may expose one separate, nonqualified observation set. Prefer an evaluable four-name set; within that count choose fewer failed gates, smaller normalized overruns, a higher historical return lower bound, and then deterministic symbol order. Its `observation_stock_sleeve_weight` values may use the ten-point grid and sum to 100% only inside the candidate observation pool. They are not total-account weights, must not populate the qualified research or action fields, and must be labelled nonqualified, nonactionable, and nonoptimal. A price observation line is not an entry permission.

Four candidates are the attention default. Before the cycle overlay, three, four, and five holdings have maximum stock-exposure caps of 70%, 80%, and 85% respectively; the cycle overlay may lower but never raise those caps. Weights are generated from inverse downside volatility with a bounded signal tilt and explicit position/industry caps. Non-deployed weight remains cash, financing is zero, and the legacy fixed 30% / 20% / 20% / 10% allocation is compatibility-only. Exposure bands remain research hypotheses until validated out of sample.

## Exact and operational weight contract

Preserve the full-precision continuous `research_weight` and `position.weight` values as total-account audit targets and as the basis for discretization. Their exact stock-sleeve shares are `research_weight / research_stock_exposure` and `position.weight / stock_exposure`, respectively. They do not drive final portfolio risk or return metrics. Never overwrite these audit targets with operational values.

The operational allocation uses ten-percentage-point increments **inside the stock sleeve**, and its sleeve shares must sum to exactly 100%. Apply these per-name sleeve bounds:

| Holdings | Per-name operational stock-sleeve share |
|---:|---:|
| 3 | 20%–50% |
| 4 | 10%–40% |
| 5 | 10%–30% |

Expose the stock-sleeve value as `operational_stock_sleeve_weight` and the corresponding total-account value as `operational_account_weight`. Convert the former to the latter by multiplying it by the stock exposure after the cycle overlay: use `research_stock_exposure` for the research allocation and `stock_exposure` for the action allocation. Cash is calculated separately as one minus stock exposure; financing remains zero. Thus a 30% stock exposure and a 40% sleeve share mean 12% of total account equity, not 40%.

For each security set, exhaustively enumerate only the ten-point grids that satisfy the per-name bounds and total-account industry structure, then select the single grid with the minimum sum of squared stock-sleeve deviations from the continuous audit target. Resolve an exact error tie deterministically in original research order. Only this nearest structurally feasible grid advances to final evaluation.

Recompute portfolio downside volatility, rolling drawdown, five-day expected shortfall, position downside-risk contribution, industry concentration, holding-period return bounds, and every other reported or hard portfolio metric using `operational_account_weight`, never the continuous target. If the selected nearest grid fails a risk budget, reject that security set. Do not search a second-nearest or farther grid to rescue it, because risk-aware grid shopping would over-deviate from the continuous target. If no security set survives, do not form a portfolio and retain cash; never relax a limit or present the continuous audit target as executable. Both exact and operational weights are research outputs under the current universe, constraints, and deterministic search approximation. They are not brokerage positions, orders, a globally optimal solution over all possible market combinations, or a future-optimal allocation.

## Horizons

| Horizon | Role |
|---|---|
| 1 week / 5 sessions | Entry timing and immediate failure detection |
| 2 weeks / 10 sessions | Short-term strength confirmed by the 20/60-session structure rather than a five-day spike |
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

1. Price cycle: the currently implemented classifier uses whole-market MA20 breadth and median 20-session return, plus core-index MA20/MA60/MA120 breadth, median 20/60-session returns, median 60-session annualized volatility, and median/worst 120-session drawdown.
2. Industry context: relative strength, breadth, valuation, and concentration. Use point-in-time industry membership for historical work.
3. Fundamentals: growth, profitability, cash conversion, balance-sheet strength, dilution, and industry-relative valuation. Treat financial and non-financial firms separately.
4. Trend and stage: Edwards–Magee primary/secondary trend, base, breakout, failed breakout, support/resistance, volume confirmation, and stage maturity.
5. Livermore execution discipline: act only at confirmed pivotal points, add only to profitable positions, avoid averaging down, and wait when evidence conflicts. Express fixed historical dollar rules with ATR/volatility-scaled thresholds.
6. Catalysts and risks: official filings first, then licensed or attributable news. Separate new information from already-priced narrative.
7. Portfolio construction: expected net return distribution, downside deviation, CVaR, maximum drawdown, Sharpe/Sortino/Calmar confidence bounds, correlation, and turnover costs.

Use a transparent deterministic rank ensemble before any language-model review. The language model may explain and challenge structured evidence but must not manufacture factor values.

## Cycle and security decision layers

Market-cycle classification and security selection are two independent layers:

1. The **cycle layer** describes the environment and sets the maximum exposure, downside-risk budget, and strictness of entry confirmation.
2. The **security layer** uses Edwards–Magee trend, base, breakout, volume, maturity, fundamentals, execution, and security-level risk evidence to rank three to five candidates.
3. The **deployment layer** combines the two. A candidate can be `CONDITIONAL_ENTRY`, `WAIT_CONFIRMATION`, or `OBSERVE_ONLY`; the actionable count may be zero. These are research labels, not orders.

The current deterministic cycle implementation is a **single-cutoff price-cycle proxy**, based exactly on the inputs listed above. Turnover, valuation, credit, investor psychology, and cross-stock correlation are not inputs to the current state classifier. It does not implement Howard Marks's complete economic, profit, credit-availability, valuation, and investor-psychology cycle. Do not label a price state as a precise economic-cycle forecast or claim that the beginning, end, timing, or amplitude of a bull or bear market is known. Its `confidence` field is rule agreement, not a probability of future market direction.

Use the five implemented, evidence-backed states plus the data-unavailable state:

| Code state | UI label | Maximum stock exposure | Entry strictness |
|---|---|---:|---|
| `UPTREND_EXPANSION` | 中期上行｜短线增强 | 80% | `STANDARD` |
| `UPTREND_PULLBACK` | 中期上行｜短线回撤或分化 | 60% | `TIGHT` |
| `TRANSITION_RECOVERY` | 中期过渡｜复苏尝试或证据混合 | 50% | `TIGHT` |
| `DOWNTREND_REPAIR` | 中期下行｜短线修复反弹 | 30% | `DEFENSIVE` |
| `DOWNTREND_PRESSURE` | 中期下行｜短线压力 | 20% | `EXCEPTION_ONLY` |
| `UNAVAILABLE` | 价格周期数据不可用 | 0% | `UNAVAILABLE` |

The current version does **not** persist state across decision dates and does not implement hysteresis. State persistence and hysteresis are possible future controls that require point-in-time validation before adoption. Cross-stock breadth and core-index evidence remain independent inputs, but a defensive state or legacy `risk_off` assessment must not terminate candidate screening. The stricter entry rules may leave every candidate waiting or observation-only, so actual deployment may remain at zero even though the usable-state exposure ceiling is 20% or more. Stale, misaligned, unavailable, or unlicensed required index evidence is a data problem and must fail closed rather than being silently dropped.

Howard Marks's cycle framework governs the balance between aggressiveness and defensiveness and the interpretation of excesses and corrections. It is not an individual-stock ranking formula or a market-timing promise. Edwards–Magee governs the security's primary/secondary trend, base, breakout, failure, support/resistance, volume confirmation, and stage maturity. Neither layer may override stale data, execution failures, weak fundamentals, late-stage acceleration, or portfolio-risk constraints. A current-only balance-sheet rank is a narrow solvency/liquidity proxy, not the complete fundamentals layer, and is forbidden in historical replay.

## Validation before claiming a production result

- Use purged walk-forward evaluation with an embargo at least as long as the decision-to-execution gap.
- Form the portfolio at each historical decision date from the universe and information genuinely known then; do not select today's four stocks and backfill them through history.
- Include delisted names, historical ST/suspension states, corporate actions, fees, tax, slippage, and unfilled entries.
- Report block-bootstrap confidence intervals for Sharpe and Calmar, non-overlapping horizon outcomes, turnover, capacity, worst fold, maximum drawdown, and CVaR.
- Validate the five implemented price-cycle states, exposure overlays, and security selection both jointly and separately. Compare the double-layer policy with an always-invested benchmark and with the former risk-off hard stop. Separately test whether adding persistence or hysteresis improves out-of-sample behavior before implementing either feature.
- Keep a research-only status until the out-of-sample thresholds are met. Never substitute in-sample metrics.
- Before that validation is complete, cycle labels, candidate ranks, exposure caps, and entry states are research outputs, not promises of future return, bear-market resilience, or successful market timing.

## Final report contract

Return either `DATA_NOT_READY`, `VALIDATION_NOT_READY`, `NO_ELIGIBLE_PORTFOLIO`, or `RESEARCH_ONLY`.

`DATA_NOT_READY` is reserved for data defects such as stale or mismatched cutoffs, insufficient required history or coverage, unknown units, missing required market evidence, or unlicensed inputs. A defensive market state is not `DATA_NOT_READY`.

Only `RESEARCH_ONLY` may contain a final actionable list of three to five securities.
`VALIDATION_NOT_READY` may expose provisional machine candidates solely for evidence review when
they are clearly marked non-actionable and `final_buy_list=false`; it must never phrase them as a
recommendation. `NO_ELIGIBLE_PORTFOLIO` means security-level gates, the current cycle-specific entry
checks, or portfolio constraints left fewer than three feasible holdings. The cycle state must not
stop the underlying candidate screen. A data-qualified report may have zero `CONDITIONAL_ENTRY`
candidates; in that case it remains cash and cannot contain a deployed `RESEARCH_ONLY` portfolio.

Every data-qualified report must show the implemented cycle label, its evidence and rule-agreement
confidence, three to five ranked candidates when at least three pass the security gates (otherwise
the available zero to two without forming a portfolio), each candidate's deployment state, the
actionable count, the exposure cap, and cash. For each security include rank, automatic weight,
thesis by horizon, entry condition, structural invalidation, reduction conditions, evidence,
principal risks, and data
confidence. For the portfolio include cash, zero borrowing, historical and out-of-sample metrics
separately, scenario ranges, evidence coverage, strategy/data versions, and the next scheduled
review condition.
