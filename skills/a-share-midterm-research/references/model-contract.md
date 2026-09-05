# A-share model contract

Production uses [continuous-signal-v1](#current-production-mode-continuous-signal-v1),
whose section below takes precedence over all legacy horizon/default/output
rules. The fixed-horizon sections are preserved for historical research only.

## Version and implementation truth

**Central implementation status:** `integration_pending`

The nested-timeframe rules below are the normative target for
`multi-timeframe-contract-v0.2.0`. Writing the
contract does not prove that a software run implements it. A run may declare this strategy version
only when its immutable archive contains equivalent evidence for completed weekly/monthly cutoffs,
per-horizon structure gates, per-horizon risk and return lower-confidence bounds, per-horizon price
levels, and cross-horizon overlap attribution. A run missing any of those items must identify itself
as `legacy_single_daily_gate` or `partial_multiframe`; it must not describe six parameterizations of
one daily screen as six independent timeframe models. Runtime archive metadata, not repository prose
or UI copy, is the source of truth.

This is the single repository-level implementation-status marker for the contract. Component-level
versions do not change it. Set it to `integrated/research_validation_pending` only after the complete
service path has passed real integration acceptance for every required archive and report semantic,
and the required purged walk-forward path has produced an auditable out-of-sample acceptance archive.
Further investment-validity thresholds may still remain pending after that transition.

**Current runtime note (2026-08-28):** the multi-timeframe analytics core is wired through the
portfolio service, UI, read-only MCP, and evening digest. A read-only integration run using the
2026-08-27 common cutoff exercised all six horizon routes. That acceptance proves component plumbing,
not the complete contract. Runtime results remain `partial_multiframe`, and the central status stays
`integration_pending`, because four requirements are still open:

1. point-in-time official-exchange-calendar boundaries for completed weekly and monthly bars;
2. one immutable per-run archive containing all six-horizon cutoffs, gates, risk/LCB, price-source,
   and overlap/difference evidence;
3. calibrated primary-weekly/monthly late-stage maturity gates; and
4. the strict purged walk-forward out-of-sample acceptance described below.

The current late-stage protection is narrower than the target contract: one shared daily
5/20/60/120-session acceleration freeze is combined with each horizon's structure state and daily
execution `EXTENDED` status. It must not be described as six independently calibrated weekly/monthly
maturity gates. Do not add unvalidated primary-timeframe overheat thresholds merely to close this gap.

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

## Nested-timeframe horizon contract

Keep four time concepts independent and archive all four:

1. **Signal sampling interval** is the daily, weekly, or monthly bar aggregation.
2. **Signal lookback** is the number of completed bars used on that sampling interval.
3. **Planned holding/evaluation horizon** is the forward session count used for outcome labels and
   the net-return lower-confidence bound.
4. **Review/rebalance cadence** determines when a portfolio may be reconstituted. A daily action
   review is not permission or a requirement to replace holdings every day.

Do not infer any one of these from another. In particular, a 60-session holding horizon does not
imply a 60-daily-bar signal, and a 252-daily-bar lookback does not imply either a one-year holding
period or monthly rebalancing.

Use slow, primary-structure, and execution timeframes as follows:

| Horizon | Slow context | Primary structure | Execution evidence | Primary / slow refresh |
|---|---|---|---|---|
| 1 week / 5 sessions | last completed weekly bars | daily fresh breakout or first healthy retest | completed daily bar | daily / weekly |
| 2 weeks / 10 sessions | last completed weekly bars | ordered daily short/intermediate base; reject five-day spikes | completed daily bar | daily / weekly |
| 1 month / 20 sessions | last completed weekly bars | daily base, volume-confirmed breakout, and post-breakout acceptance | completed daily bar | daily / weekly |
| 3 months / 60 sessions | last completed monthly bars | completed weekly trend, base, and breakout/retest | daily confirmation and executable condition | weekly / monthly |
| 6 months / 120 sessions | last completed monthly bars | completed weekly primary trend and secondary breakout/retest | weekly confirmation plus daily executability | weekly / monthly |
| 1 year / 252 sessions | last completed monthly primary trend | completed weekly long base and continuation breakout | weekly confirmation plus daily executability | weekly / monthly |

The deterministic analytics core starts with the following **unvalidated v0.2 research defaults**.
`D`, `W`, and `M` mean completed daily, weekly, and monthly bars. These values distinguish signal
lookbacks from holding horizons; they are not empirically optimal, permanent, or permission to tune
against the full sample. Any change requires a method-version change and new point-in-time validation.

| Horizon | Slow fast/slow MA | Primary lookback/recent/base | Minimum daily history |
|---|---:|---:|---:|
| 1 week | W4/W13 | D20/D5/D12 | 80 sessions |
| 2 weeks | W4/W13 | D30/D8/D18 | 95 sessions |
| 1 month | W8/W26 | D60/D10/D30 | 140 sessions |
| 3 months | M3/M6 | W13/W4/W8 | 180 sessions |
| 6 months | M6/M12 | W26/W6/W13 | 280 sessions |
| 1 year | M12/M24 | W52/W8/W26 | 540 sessions |

| Horizon | Daily execution breakout/recent/MA | Daily anchor MA | Relative-strength sessions |
|---|---:|---:|---:|
| 1 week | D20/D5/D10 | D20 | 5 |
| 2 weeks | D30/D8/D10 | D20 | 10 |
| 1 month | D40/D10/D20 | D60 | 20 |
| 3 months | D60/D15/D20 | D120 | 60 |
| 6 months | D90/D20/D40 | D120 | 120 |
| 1 year | D120/D25/D50 | D252 | 252 |

This is a nested model, not an exclusive rule that short horizons use only daily charts and long
horizons use only monthly charts. The slow timeframe permits direction, the primary timeframe
qualifies the base/breakout/retest, and the execution timeframe translates an already-qualified
structure into a next-session condition and invalidation. Daily execution does not equal daily
selection. For horizons of three months or longer, daily evidence must never override a failed or
missing completed weekly/monthly gate.

Weekly and monthly bars must be complete as of the decision cutoff. During a week or month, the
partial aggregate may be retained only as explicitly identified execution evidence; it cannot pass a
weekly or monthly hard gate. A holiday-shortened week becomes complete after the final actual exchange
session of that week. Bar construction must use the point-in-time exchange calendar, not a fixed
five-sessions-per-week approximation.

If the official calendar is unavailable, a `W-FRI`/business-month-end fallback may conservatively
defer a possibly complete shortened period, but it must never close one early. Archive the fallback
and its reason, label contract implementation partial, and do not claim full integration until the
official-calendar completed-period boundary is used.

Require adjacent-horizon coherence, but do not require different names for visual variety. The same
security may legitimately pass several horizons when each horizon independently confirms its own
slow context, primary structure, risk budget, return LCB, entry condition, and invalidation. Such a
result is genuine confluence. Every six-horizon report must show a pairwise overlap rate or overlap
matrix and per-security attribution explaining whether overlap arose from the shared eligibility gate,
independent structure confirmation, or merely similar ranking. It must also explain why a security
entered one horizon but not an adjacent horizon. Without independent gates and this attribution, do
not call overlap “multi-timeframe confirmation.”

Candidate overlap uses pairwise Jaccard similarity, `|A intersection B| / |A union B|`; report it
separately for machine candidates and risk-qualified portfolios. If both sets are empty, report the
value as unavailable rather than 100%. Per-security difference attribution must distinguish at least
shared-eligibility-only, independent-structure-confluence, slow-context failure, primary-structure
failure, risk/LCB failure, untriggered price condition, and rank-only similarity. Weighted sleeve
overlap may be added, but it must not replace the set-based audit.

### Shared and horizon-specific computation

The following may be shared exactly once across horizons:

- licensed-source and common-cutoff checks;
- security identity, historical listing status, ST/delisting/suspension status, corporate-action
  integrity, liquidity, capacity, and decision-date executability;
- point-in-time financial/announcement veto evidence whose publication gate is horizon-independent.

After that shared eligibility layer, each horizon must independently compute and archive:

- slow-context and primary-structure gate results with bar cutoffs and reason codes;
- stage maturity, extension, ATR/volatility, and volume confirmation on its configured sampling
  intervals and lookbacks;
- downside-risk budget and portfolio metrics appropriate to that holding horizon;
- forward net-return distribution and lower-confidence bound using that horizon's non-overlapping or
  purged outcome labels;
- conditional entry, structural invalidation, and staged-reduction levels with their source timeframe;
- candidate ranking and the complete 3/4/5-security portfolio search using that horizon's operational
  weights.

Do not reuse a 60-session breakout line, one risk evaluation, one portfolio, or one price plan across
all six horizons and relabel it. A shared eligible universe is intentional; a shared post-eligibility
decision is not.

## Persistent holding-management contract

Recommendation generation and management of an already-held portfolio are separate state machines.
The current holding set, entry date, planned horizon, and user-supplied cost or weights come only from
an explicit user statement. They remain effective until the user explicitly replaces or clears that
statement. A new candidate ranking, an `EXIT` research label, or a newly generated model portfolio must
never silently rewrite the holding ledger.

For each verified completed close, review every active holding under its declared horizon and return one
of `HOLD`, `TIGHTEN`, `REDUCE`, `EXIT`, or `REVIEW`:

- `HOLD` keeps an intact position and ignores ordinary rank churn;
- `TIGHTEN` keeps the position while ratcheting a confirmed protection line upward;
- `REDUCE` is a next-tradable-session staged-reduction review after completed-timeframe deterioration;
- `EXIT` requires a completed close below the effective protection line; and
- `REVIEW` is fail-closed whenever price, calendar, company-action, or other required evidence cannot
  support a destructive action.

The implemented protection line is a versioned **Edwards--Magee-inspired research default**, not a claim
of reproducing every rule in a specific book edition. For the position's primary structure timeframe,
use only a reaction low confirmed with right-hand bars after entry; until one exists, use the structure
floor that was knowable at entry. Apply the documented ATR buffer and persist
`max(previous_effective_stop, new_candidate_stop)`. The effective line may never move down. A database
constraint must independently enforce that monotonic rule. A close below the line is a research trigger
for the next tradable session, not a guaranteed fill; A-share T+1, suspension, limit-down, and gaps still
apply.

The separate `holding-stop-shadow-runner-v0.1.0` experiment does not change that production default.
Its `magee-shadow-daily-v0.1.0` engine evaluates `three_day_escape_6pct` and
`new_high_3pct_6pct` independently, scanning only complete daily bars from the recorded entry date through
the point-in-time cutoff; pre-entry bars cannot form an anchor, candidate, or trigger. A low is confirmed
only after three consecutive
verified sessions whose complete ranges escape above the candidate day's high, and a 3%-new-high candidate
whose low becomes the next baseline. Both use a 6% buffer, never move a remembered line down, and become
effective only on the next observed daily bar. The pure engine cannot itself prove that input rows form
the complete official trading-session chain; the orchestration evidence must establish that before an
event is evaluation-eligible. The current runner has not integrated that evidence, so it archives the
counterfactuals while keeping `evaluation_eligible=0`. They write only `holding_stop_shadow_events` with
`decision_layer=shadow_research_only`, `production_decision_input=0`, `external_delivery_allowed=0`, and
`auto_order_allowed=0`; they cannot update production stops, holding actions, notifications, or orders.
Missing or stale interval-complete company-action clearance makes an event non-comparable. The shadow
schema records coverage-start, clearance-through, and knowledge-time fields, but the existence of those
fields is not evidence: if the upstream source cannot supply the complete interval and point-in-time
knowledge boundary, `evaluation_eligible` must remain zero. Runs that
depend on the current `W-FRI`/business-month-end fallback remain partial until point-in-time official-calendar
boundaries are implemented. No shadow version may be promoted without per-horizon paired purged walk-forward
evidence, costs and A-share executability, multiplicity-adjusted intervals, at least 30 valid paired paths
across three non-overlapping folds and two market states, explicit user approval, and a new production method
version. These gates do not assert that the shadow rules are superior to the incumbent.

Unadjusted prices cannot distinguish genuine trend failure from every ex-rights, dividend, bonus-share,
split, or rights event. A missing or stale independent company-action clearance must therefore block a
new protection-line write and convert apparent `TIGHTEN`, `REDUCE`, or `EXIT` into urgent `REVIEW`.
An unambiguously detected action also requires review. A non-destructive `HOLD` may be shown without a
clearance, but it must not advance the remembered protection line. Never treat a previous-candle close as
an exchange ex-right reference.

Pruning is deliberately asymmetric: keep strong or intact branches, never average down, and respond to
confirmed weakness rather than every red day. A reduction or exit leaves the released weight in cash.
The model does not automatically fill the slot, rebalance the remaining positions, or place an order.
Adding a replacement requires a separately generated data-qualified plan and an explicit user holding
update. This avoids turning the fruit-tree metaphor into daily high-turnover rank chasing.

The complete holding review and protection-line history stay local. External summaries may include only
the minimum explicitly authorized fields and only for specifically authorized channels; absence of an
allow-list is denial, not implied consent. Costs, total account amounts, raw history, and evidence payloads
must not be placed in notification bodies or sanitized scheduler logs.

## Rolling performance and monthly learning contract

Every published six-horizon cohort freezes its plan date, method version, structured price condition, and
operational weights before provider submission. Settle the 5/10/20/60/120/252-session outcomes on the
verified trading-session chain without rewriting the prediction, renormalizing unentered weights, or
mixing formal action, original observation, and reconstructed observation populations. Unadjusted price
return excludes dividends and is not account total return. Missing company-action coverage, an incomplete
entry rule, suspension, or a missing due-date mark remains pending or needs review rather than becoming a
fabricated return.

Once per completed calendar month, summarize each population and horizon independently. Filter every
mutable result by the knowledge time of the review, require an adequate count of verifiable matured
batches before a directional conclusion, and report a market comparison only when a same-interval,
point-in-time benchmark with matching cutoff and adjustment is supplied. A truncated archive scan or an
unavailable benchmark must be stated explicitly.

Weak or negative outcomes may create a versioned experiment proposal and an attribution checklist; they
must never mutate thresholds automatically. Any proposed change requires user approval, a new strategy
version, and a fresh purged walk-forward comparison against the frozen incumbent before it may become the
production default.

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

The late-acceleration bullet above is the normative target. In the current partial runtime, only the
shared daily acceleration freeze plus horizon structure/daily `EXTENDED` evidence is implemented;
weekly/monthly primary-timeframe maturity thresholds remain pending calibration.

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
- Validate all six slow/primary/execution timeframe mappings separately. Rebuild completed weekly and
  monthly bars at every historical decision date from the then-known exchange calendar, including
  holiday-shortened weeks, and prove that partial higher-timeframe bars never pass a gate.
- Ablate sampling interval, signal lookback, holding/evaluation horizon, and review/rebalance cadence
  independently so that apparent performance cannot come from conflating those four concepts.
- Compare `multi-timeframe-contract-v0.2.0` with `legacy_single_daily_gate` under identical point-in-time universe,
  costs, slippage, capacity, and portfolio constraints. Report the incremental result, turnover,
  worst fold, and a multiplicity-adjusted confidence interval rather than only the new model in sample.
- Evaluate horizon overlap jointly. The same security across several horizons is correlated evidence,
  not several independent observations; never multiply confidence or count it repeatedly in a pooled
  significance test. Report false-confluence and neighboring-horizon disagreement diagnostics.
- Keep a research-only status until the out-of-sample thresholds are met. Never substitute in-sample metrics.
- Before that validation is complete, cycle labels, candidate ranks, exposure caps, entry states, and
  multi-timeframe confluence are research outputs, not promises of future return, bear-market resilience,
  successful market timing, or superiority over the daily baseline.

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
thesis by horizon, entry condition and its source timeframe, structural invalidation, reduction conditions, evidence,
principal risks, and data
confidence. For the portfolio include cash, zero borrowing, historical and out-of-sample metrics
separately, scenario ranges, evidence coverage, strategy/data versions, and the next scheduled
review condition. A report declaring `multi-timeframe-contract-v0.2.0` must additionally show the last completed
daily/weekly/monthly bar cutoffs, per-horizon structure/risk/LCB status, pairwise horizon overlap, and
per-security difference attribution. The exact serialized field names may evolve, but these audit
semantics may not be omitted.
# Current production mode: continuous-signal-v1

This section supersedes the **horizon/default/output** provisions of the legacy
contract below; its evidence, privacy, point-in-time and no-order safeguards
still apply. Do not silently reactivate six-horizon production recommendations.

- One ongoing 3–5-stock research target, no forced holding deadline. Fewer
  remaining positions or cash are valid; do not relax risk gates to fill slots.
- Daily/completed-weekly signal windows are independently frozen. The old
  `holding_weeks=4` transport field is not a sell date. Protect existing stops
  from downward movement, including same-day recomputation/model migration.
- Every new entry requires confirmed EARLY_UPTREND plus base breakout/healthy
  retest and execution/evidence/risk gates. The 120-session <=15% early-stage
  threshold is an unvalidated hypothesis, not absolute-bottom identification.
  Do not reapply new-entry eligibility to intact existing holdings.
- Keep the structural-stop/maximum-entry-price gate: prospective initial loss
  distance <=8%, not a realized-loss guarantee. Unknown corporate-action
  coverage (from entry through cutoff) blocks confirmed exit/stop persistence.
- Lock actual retained weights and compare every admitted replacement plus
  cash. Initial top36/beam128 search is approximate; single replacement is
  enumerated on 10% total-account new-allocation steps. Do not round old weights.
  One name per industry, <=30% total-account single-name cap and original risk
  constraints remain binding; 1–2 remaining names do not waive risk-contribution limits.
- Require an explicit whole-account snapshot and verified price/corporate-action
  interval for drifted weights. No recommendation constitutes a confirmed sale,
  purchase, deposit or withdrawal. Pending exits are local contingencies only.
- Rank with the frozen historical 20-session return-LCB proxy; do not claim
  future/global optimum Sharpe or validated dynamic-strategy returns. New
  continuous decision/NAV records are separate from immutable legacy maturities.
  Actual versus shadow, cash, open positions, costs and company actions must be
  distinguished; absent evidence is unavailable, not zero return.
- Only authorized ServerChan summaries/images may leave the local machine.
  Verify exact holding version before disclosure; R2 secrets and raw financial
  data are never report content. Provider acceptance is not end-device delivery.

The remainder is the preserved fixed-horizon research contract, used only for
explicit legacy comparisons and interpreting frozen historical archives.
