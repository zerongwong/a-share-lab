# Changelog

All notable changes are recorded here. The project follows semantic versioning for software releases;
strategy and data-schema versions are tracked separately in research archives.

## [Unreleased]

- Add an isolated `holding-stop-shadow-runner-v0.1.0` research track for a strict daily-bar
  Edwards--Magee translation. The `magee-shadow-daily-v0.1.0` engine keeps
  `three_day_escape_6pct` and `new_high_3pct_6pct` separate, scans only entry-date-through-cutoff
  evidence, and uses a frozen 6% buffer. Shadow events live only in `holding_stop_shadow_events`, are
  not production decision inputs, cannot notify or place orders, and remain ineligible for promotion until point-in-time
  company-action/calendar, A-share execution, paired purged walk-forward, and multiplicity-adjusted
  acceptance gates pass. This does not claim superiority over the current ATR-based protection line.
- Add an opt-in private current-holding chart report for the four-name, one-month case: one pastel
  4-by-2 composite PNG with a daily and completed-weekly panel per holding, A-share red-up/green-down
  candles, model observation conditions, moving averages, volume, confirmed references, and effective
  protection lines. Reconstructed observations remain explicitly non-actionable; lines begin only when
  knowable, incomplete weeks are excluded, and costs, shares, amounts, and account weights are omitted.
  Generation requires channel-specific holding-summary authorization; local archives use private
  permissions and a 30-day retention window. ServerChan can receive one optional image through a private
  Cloudflare R2 object, a signed URL capped at one hour, and a separately confirmed one-day bucket
  lifecycle; Bark never receives the image. Publication requires an exact holding revision plus separate
  summary/chart/provider grants, repeats those checks before and after upload and immediately before
  provider submission, and revokes the object when authorization changes or ServerChan does not accept
  the image. Missing configuration still falls back to concise text, and no chart is uploaded to GitHub
  or a public image host.
- Add a private, explicit current-holding ledger and close-confirmed fruit-tree review. Holdings and
  horizons persist until the user replaces or clears them; daily reviews emit HOLD/TIGHTEN/REDUCE/
  EXIT/REVIEW without changing membership or placing orders. The Edwards--Magee-inspired protection
  line uses confirmed post-entry pivots, is enforced never to move down, and requires independent
  company-action clearance before destructive actions or stop ratchets. The daily scheduler stores only
  aggregate action counts; channel-specific holding delivery remains deny-by-default.
- Add a private monthly model-review archive after the first verified session of each new month. Formal
  action, original observation, and reconstructed observation remain separate across all six horizons;
  mutable results are filtered by knowledge time, insufficient/truncated samples do not create conclusions,
  and an unavailable same-interval point-in-time benchmark is reported as unavailable. Reviews may propose
  versioned experiments, but never change the production model without user approval and a fresh purged
  walk-forward comparison.
- Add immutable six-horizon recommendation archives and idempotent maturity-settlement components.
  Maturity is measured 5/10/20/60/120/252 verified sessions after `plan_for_date`; the 15:30/18:30
  daily-sync path invokes settlement only after a verified close, while the independent 21:00 task
  remains a new-plan publisher. Results are separated into official action, original observation,
  and reconstructed-observation populations. They use unadjusted price returns, exclude dividends,
  preserve published weights without reweighting, and do not automatically backfill legacy reports.
  Corporate-action coverage stays unknown unless an independent point-in-time action or adjustment-factor
  source is injected; prior-candle closes are never treated as exchange ex-right references. Historical
  reconstruction is explicitly a current-model replay, not a recovery of the original software payload.
- Document an out-of-sample `U0–U3 × C0–C2` industry/leader experiment instead of adopting a fixed
  leader universe. The proposed secondary-industry maximum of one stock, primary-industry maximum of
  two stocks, and primary-industry sleeve cap of 40% remain an unvalidated candidate default rather
  than an enabled production rule.
- Fix the macOS evening-report weekday mapping so Sunday-to-Thursday uses launchd weekdays
  `0,1,2,3,4`; the former `1..5` mapping skipped Sunday plans. Also stop a no-session weekend
  sync before requesting the full stock universe, avoiding unnecessary provider failures.
- Wire the multi-timeframe analytics core through the portfolio service, Streamlit UI, read-only MCP,
  and evening digest. A read-only six-horizon acceptance run on the 2026-08-27 common cutoff confirmed
  that the runtime routes are connected; it did not validate future performance or complete the central
  contract. Central status remains `integration_pending`, and runtime output remains
  `partial_multiframe`, until point-in-time official-calendar weekly/monthly boundaries, an immutable
  six-horizon per-run archive, calibrated weekly/monthly late-stage maturity gates, and strict purged
  walk-forward acceptance are complete.
- Document the implemented late-stage boundary accurately: the runtime uses one shared daily
  5/20/60/120-session acceleration freeze plus horizon structure and daily `EXTENDED` evidence. It does
  not yet claim independently calibrated weekly/monthly maturity gates, and it does not add unvalidated
  primary-timeframe overheat thresholds.
- Define the `multi-timeframe-contract-v0.2.0` target contract: separate signal sampling, lookback, holding horizon,
  and rebalance cadence; use completed weekly/monthly bars; and give all six horizons independent
  structure gates, risk budgets, return LCBs, and price plans after one shared eligibility gate. A run
  may claim this version only when its archive also contains horizon-overlap and difference attribution;
  this documentation entry does not itself claim that implementation or out-of-sample validation is complete.
- Define a price-cycle-aware medium-term research contract: identify early or orderly main uptrends,
  require explicit entry structure, and use the market cycle to adjust confirmation strictness and risk
  budget rather than stopping candidate discovery.
- Separate the 3–5-stock machine candidate layer from the actionable layer. Risk-qualified plans retain
  account-level exposure and price conditions; rejected near-miss combinations can only appear as clearly
  labelled observation plans and never inherit actionable weights.
- Replace awkward continuous operating weights with a deterministic 10%-grid stock-sleeve allocation that
  sums to 100%, while retaining continuous research weights for audit and risk comparison.
- Add a verified, source-isolated daily market overlay for licensed end-of-day price and volume increments.
  The optional macOS schedule can run an initial 15:30 sync and an idempotent 18:30 quality recheck without
  modifying the read-only CSMAR history baseline.
- Add the 2-week horizon (10 trading sessions), completing the six research horizons: 1 week, 2 weeks,
  1 month, 3 months, 6 months, and 1 year.
- Add an optional independent 21:00 Server酱 evening report task. It summarizes the market price-cycle
  posture and all six horizons for the next trading day, deduplicates by verified cutoff and method version,
  and reads its SendKey only from macOS Keychain.
- Add local Server酱 and Bark credential/test adapters backed by macOS Keychain, plus independent CI,
  data-policy, architecture, and contribution templates.
- Keep the entire workflow research-only: no brokerage connection, account-position access, automatic
  order placement, or claim of future return probability.

## [0.1.0] - 2026-08-26

- Initial local CSMAR import, four-stock research prototype, Streamlit UI, read-only MCP, and immutable research archive.
