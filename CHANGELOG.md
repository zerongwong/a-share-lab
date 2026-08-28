# Changelog

All notable changes are recorded here. The project follows semantic versioning for software releases;
strategy and data-schema versions are tracked separately in research archives.

## [Unreleased]

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
