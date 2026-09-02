PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

-- An evening report is the immutable statement that was generated for a
-- future trading session.  ``archive_nature`` keeps a later reconstruction
-- visibly separate from an original, provider-submitted report.
CREATE TABLE IF NOT EXISTS recommendation_reports (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    archive_nature TEXT NOT NULL CHECK (
        archive_nature IN ('original', 'reconstructed')
    ),
    decision_date TEXT NOT NULL,
    plan_for_date TEXT NOT NULL,
    common_cutoff TEXT NOT NULL,
    method_version TEXT NOT NULL,
    cycle_label TEXT NOT NULL,
    entry_strictness TEXT NOT NULL,
    max_stock_exposure REAL NOT NULL CHECK (max_stock_exposure BETWEEN 0.0 AND 1.0),
    minimum_cash_weight REAL NOT NULL CHECK (minimum_cash_weight BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    CHECK (common_cutoff <= plan_for_date)
);

-- Every horizon is a separate cohort.  Initial archive status is immutable;
-- maturity state belongs only in recommendation_batch_results.
CREATE TABLE IF NOT EXISTS recommendation_batches (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES recommendation_reports(id) ON DELETE CASCADE,
    horizon_key TEXT NOT NULL CHECK (
        horizon_key IN ('1w', '2w', '1m', '3m', '6m', '1y')
    ),
    holding_weeks INTEGER NOT NULL,
    holding_sessions INTEGER NOT NULL,
    label TEXT NOT NULL,
    data_cutoff TEXT,
    source_status TEXT NOT NULL,
    evaluation_mode TEXT NOT NULL CHECK (
        evaluation_mode IN (
            'action_simulation', 'observation_simulation',
            'reconstructed_observation', 'unavailable'
        )
    ),
    cohort_nature TEXT NOT NULL CHECK (
        cohort_nature IN (
            'action_qualified', 'risk_qualified', 'observation_only', 'unavailable'
        )
    ),
    stock_exposure REAL NOT NULL CHECK (stock_exposure BETWEEN 0.0 AND 1.0),
    cash_weight REAL NOT NULL CHECK (cash_weight BETWEEN 0.0 AND 1.0),
    anchor_session_date TEXT,
    member_count INTEGER NOT NULL CHECK (member_count >= 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'unavailable')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (report_id, holding_sessions),
    CHECK (
        (holding_weeks = 1 AND holding_sessions = 5 AND horizon_key = '1w') OR
        (holding_weeks = 2 AND holding_sessions = 10 AND horizon_key = '2w') OR
        (holding_weeks = 4 AND holding_sessions = 20 AND horizon_key = '1m') OR
        (holding_weeks = 13 AND holding_sessions = 60 AND horizon_key = '3m') OR
        (holding_weeks = 26 AND holding_sessions = 120 AND horizon_key = '6m') OR
        (holding_weeks = 52 AND holding_sessions = 252 AND horizon_key = '1y')
    )
);

CREATE TABLE IF NOT EXISTS recommendation_members (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES recommendation_batches(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL CHECK (rank > 0),
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    action TEXT NOT NULL,
    allocation_nature TEXT NOT NULL CHECK (
        allocation_nature IN (
            'action_research', 'risk_qualified_research', 'observation_only', 'unavailable'
        )
    ),
    stock_sleeve_weight REAL CHECK (stock_sleeve_weight BETWEEN 0.0 AND 1.0),
    account_weight REAL CHECK (account_weight BETWEEN 0.0 AND 1.0),
    price_nature TEXT NOT NULL CHECK (
        price_nature IN ('conditional_entry', 'observation_only', 'unavailable')
    ),
    plan_kind TEXT,
    price_low REAL CHECK (price_low IS NULL OR price_low >= 0.0),
    price_high REAL CHECK (price_high IS NULL OR price_high >= 0.0),
    trigger_price REAL CHECK (trigger_price IS NULL OR trigger_price >= 0.0),
    reference_price REAL CHECK (reference_price IS NULL OR reference_price >= 0.0),
    observation_anchor TEXT NOT NULL DEFAULT 'none' CHECK (
        observation_anchor IN (
            'none', 'plan_session_close', 'archived_reference_price'
        )
    ),
    confirmation_rule TEXT,
    invalidation_price REAL CHECK (invalidation_price IS NULL OR invalidation_price >= 0.0),
    plan_cutoff TEXT,
    plan_sessions INTEGER CHECK (plan_sessions IS NULL OR plan_sessions > 0),
    plan_method_version TEXT,
    price_condition TEXT NOT NULL,
    evidence_pending INTEGER NOT NULL DEFAULT 0 CHECK (evidence_pending IN (0, 1)),
    primary_timeframe TEXT,
    primary_structure TEXT,
    entry_plan_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (batch_id, symbol),
    UNIQUE (batch_id, rank),
    CHECK (price_low IS NULL OR price_high IS NULL OR price_low <= price_high)
);

-- Delivery records are append-only.  Provider acceptance is intentionally not
-- described as end-device delivery confirmation.
CREATE TABLE IF NOT EXISTS recommendation_delivery_events (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES recommendation_reports(id) ON DELETE CASCADE,
    batch_id TEXT REFERENCES recommendation_batches(id) ON DELETE CASCADE,
    delivery_kind TEXT NOT NULL,
    channel TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    provider_status TEXT NOT NULL,
    provider_receipt_id TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

-- Results are intentionally mutable/idempotent observations.  They never
-- rewrite the recommendation rows above.  A missing/not-entered member is not
-- represented as a zero return.
CREATE TABLE IF NOT EXISTS recommendation_member_results (
    id TEXT PRIMARY KEY,
    member_id TEXT NOT NULL UNIQUE REFERENCES recommendation_members(id) ON DELETE CASCADE,
    evaluated_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'resolved', 'not_entered', 'unavailable', 'needs_review')
    ),
    entry_date TEXT,
    entry_price REAL CHECK (entry_price IS NULL OR entry_price >= 0.0),
    maturity_date TEXT,
    maturity_close REAL CHECK (maturity_close IS NULL OR maturity_close >= 0.0),
    realized_return REAL,
    holding_sessions_observed INTEGER CHECK (
        holding_sessions_observed IS NULL OR holding_sessions_observed >= 0
    ),
    max_drawdown REAL,
    max_runup REAL,
    reason_code TEXT,
    company_action_clear INTEGER CHECK (company_action_clear IN (0, 1)),
    data_cutoff TEXT,
    method_version TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_batch_results (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL UNIQUE REFERENCES recommendation_batches(id) ON DELETE CASCADE,
    evaluated_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'partial', 'resolved', 'no_entries', 'unavailable', 'needs_review')
    ),
    maturity_date TEXT,
    stock_sleeve_return REAL,
    account_return REAL,
    entered_stock_sleeve_weight REAL CHECK (
        entered_stock_sleeve_weight IS NULL OR
        entered_stock_sleeve_weight BETWEEN 0.0 AND 1.0
    ),
    entered_account_weight REAL CHECK (
        entered_account_weight IS NULL OR entered_account_weight BETWEEN 0.0 AND 1.0
    ),
    cash_weight REAL CHECK (cash_weight IS NULL OR cash_weight BETWEEN 0.0 AND 1.0),
    resolved_member_count INTEGER CHECK (
        resolved_member_count IS NULL OR resolved_member_count >= 0
    ),
    total_member_count INTEGER CHECK (
        total_member_count IS NULL OR total_member_count >= 0
    ),
    reason_code TEXT,
    data_cutoff TEXT,
    method_version TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    CHECK (
        resolved_member_count IS NULL OR total_member_count IS NULL OR
        resolved_member_count <= total_member_count
    )
);

CREATE INDEX IF NOT EXISTS idx_recommendation_reports_plan_date
    ON recommendation_reports(plan_for_date, common_cutoff);
CREATE INDEX IF NOT EXISTS idx_recommendation_batches_report_horizon
    ON recommendation_batches(report_id, holding_sessions);
CREATE INDEX IF NOT EXISTS idx_recommendation_members_batch_rank
    ON recommendation_members(batch_id, rank);
CREATE INDEX IF NOT EXISTS idx_recommendation_delivery_report_status
    ON recommendation_delivery_events(report_id, provider_status);
CREATE INDEX IF NOT EXISTS idx_recommendation_member_results_status
    ON recommendation_member_results(status, member_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_batch_results_status
    ON recommendation_batch_results(status, batch_id);

CREATE TRIGGER IF NOT EXISTS immutable_recommendation_reports_update
BEFORE UPDATE ON recommendation_reports BEGIN
    SELECT RAISE(ABORT, 'recommendation reports are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_reports_delete
BEFORE DELETE ON recommendation_reports BEGIN
    SELECT RAISE(ABORT, 'recommendation reports are immutable; retain the audit trail');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_batches_update
BEFORE UPDATE ON recommendation_batches BEGIN
    SELECT RAISE(ABORT, 'recommendation batches are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_batches_delete
BEFORE DELETE ON recommendation_batches BEGIN
    SELECT RAISE(ABORT, 'recommendation batches are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_members_update
BEFORE UPDATE ON recommendation_members BEGIN
    SELECT RAISE(ABORT, 'recommendation members are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_members_delete
BEFORE DELETE ON recommendation_members BEGIN
    SELECT RAISE(ABORT, 'recommendation members are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_delivery_update
BEFORE UPDATE ON recommendation_delivery_events BEGIN
    SELECT RAISE(ABORT, 'recommendation delivery events are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_recommendation_delivery_delete
BEFORE DELETE ON recommendation_delivery_events BEGIN
    SELECT RAISE(ABORT, 'recommendation delivery events are append-only');
END;

INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
