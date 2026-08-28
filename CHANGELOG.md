# Changelog

All notable changes are recorded here. The project follows semantic versioning for software releases;
strategy and data-schema versions are tracked separately in research archives.

## [Unreleased]

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
