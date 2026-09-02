PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

-- Shadow observations are an append-only research experiment.  They are kept
-- deliberately separate from both the production holding decision events and
-- the single mutable production protective-stop state.
CREATE TABLE IF NOT EXISTS holding_stop_shadow_events (
    id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL
        REFERENCES holding_portfolio_revisions(id) ON DELETE RESTRICT,
    position_id TEXT NOT NULL
        REFERENCES holding_positions(id) ON DELETE RESTRICT,
    position_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    holding_weeks INTEGER NOT NULL CHECK (
        holding_weeks IN (1, 2, 4, 13, 26, 52)
    ),
    holding_version INTEGER NOT NULL CHECK (holding_version > 0),
    entry_date TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    archive_nature TEXT NOT NULL DEFAULT 'live_shadow' CHECK (
        archive_nature IN ('live_shadow', 'backfill_shadow')
    ),
    variant_key TEXT NOT NULL CHECK (length(trim(variant_key)) > 0),
    status TEXT NOT NULL CHECK (
        status IN ('ready', 'no_confirmed_baseline', 'data_not_ready', 'needs_review')
    ),
    source_timeframe TEXT,
    baseline_kind TEXT,
    baseline_date TEXT,
    confirmation_date TEXT,
    baseline_price REAL CHECK (baseline_price IS NULL OR baseline_price > 0.0),
    latest_close REAL CHECK (latest_close IS NULL OR latest_close > 0.0),
    latest_low REAL CHECK (latest_low IS NULL OR latest_low > 0.0),
    candidate_stop REAL CHECK (candidate_stop IS NULL OR candidate_stop > 0.0),
    previous_shadow_stop REAL CHECK (
        previous_shadow_stop IS NULL OR previous_shadow_stop > 0.0
    ),
    effective_shadow_stop REAL CHECK (
        effective_shadow_stop IS NULL OR effective_shadow_stop > 0.0
    ),
    effective_from_date TEXT,
    next_effective_shadow_stop REAL CHECK (
        next_effective_shadow_stop IS NULL OR next_effective_shadow_stop > 0.0
    ),
    latest_intraday_touch_observed INTEGER CHECK (
        latest_intraday_touch_observed IN (0, 1)
    ),
    intraday_touch_observed INTEGER CHECK (intraday_touch_observed IN (0, 1)),
    intraday_touch_date TEXT,
    latest_close_breach_observed INTEGER CHECK (
        latest_close_breach_observed IN (0, 1)
    ),
    close_breach_observed INTEGER CHECK (close_breach_observed IN (0, 1)),
    close_breach_date TEXT,
    company_action_clear INTEGER CHECK (company_action_clear IN (0, 1)),
    company_action_evidence_id TEXT,
    company_action_evidence_source TEXT,
    company_action_covered_from TEXT,
    company_action_clear_through TEXT,
    company_action_knowledge_time TEXT,
    evaluation_eligible INTEGER NOT NULL DEFAULT 0 CHECK (evaluation_eligible IN (0, 1)),
    decision_layer TEXT NOT NULL DEFAULT 'shadow_research_only' CHECK (
        decision_layer = 'shadow_research_only'
    ),
    production_decision_input INTEGER NOT NULL DEFAULT 0 CHECK (
        production_decision_input = 0
    ),
    external_delivery_allowed INTEGER NOT NULL DEFAULT 0 CHECK (
        external_delivery_allowed = 0
    ),
    auto_order_allowed INTEGER NOT NULL DEFAULT 0 CHECK (auto_order_allowed = 0),
    parameters_json TEXT NOT NULL DEFAULT '{}',
    reason_json TEXT NOT NULL DEFAULT '[]',
    input_data_hash TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    parameter_hash TEXT NOT NULL,
    method_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (
        position_key, holding_weeks, data_cutoff,
        variant_key, method_version, parameter_hash, evidence_hash
    ),
    CHECK (
        previous_shadow_stop IS NULL OR effective_shadow_stop IS NULL OR
        effective_shadow_stop >= previous_shadow_stop
    )
);

CREATE INDEX IF NOT EXISTS idx_holding_shadow_cutoff_variant
    ON holding_stop_shadow_events(data_cutoff, variant_key, status);
CREATE INDEX IF NOT EXISTS idx_holding_shadow_position_history
    ON holding_stop_shadow_events(
        position_key, holding_weeks, variant_key, method_version, data_cutoff
    );

CREATE TRIGGER IF NOT EXISTS immutable_holding_shadow_events_update
BEFORE UPDATE ON holding_stop_shadow_events BEGIN
    SELECT RAISE(ABORT, 'holding shadow events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_holding_shadow_events_delete
BEFORE DELETE ON holding_stop_shadow_events BEGIN
    SELECT RAISE(ABORT, 'holding shadow events are immutable');
END;

INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (4, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
