PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL CHECK (
        run_type IN ('stock_analysis', 'weekly_portfolios', 'limit_watchlist')
    ),
    as_of TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    created_at TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_id TEXT,
    config_hash TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'partial', 'failed')),
    warning_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS data_snapshots (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    symbol TEXT,
    first_at TEXT,
    last_at TEXT,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    adjustment TEXT NOT NULL CHECK (
        adjustment IN ('none', 'qfq', 'hfq', 'point_in_time', 'not_applicable')
    ),
    unit_json TEXT NOT NULL DEFAULT '{}',
    checksum TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0 CHECK (is_stale IN (0, 1))
);

CREATE TABLE IF NOT EXISTS stock_analyses (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    horizon_sessions INTEGER NOT NULL CHECK (horizon_sessions > 0),
    trend_state TEXT NOT NULL,
    action_for_empty TEXT NOT NULL,
    action_for_holder TEXT NOT NULL,
    entry_low REAL,
    entry_high REAL,
    add_above REAL,
    reduce_low REAL,
    reduce_high REAL,
    invalidation REAL,
    confidence TEXT NOT NULL CHECK (confidence IN ('low', 'medium', 'high', 'unavailable')),
    rationale_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (run_id, symbol, horizon_sessions),
    CHECK (entry_low IS NULL OR entry_high IS NULL OR entry_low <= entry_high),
    CHECK (reduce_low IS NULL OR reduce_high IS NULL OR reduce_low <= reduce_high)
);

CREATE TABLE IF NOT EXISTS scenarios (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES stock_analyses(id) ON DELETE CASCADE,
    label TEXT NOT NULL CHECK (label IN ('up', 'sideways', 'down')),
    probability_low REAL,
    probability_mid REAL,
    probability_high REAL,
    return_p10 REAL,
    return_p50 REAL,
    return_p90 REAL,
    sample_n INTEGER NOT NULL DEFAULT 0 CHECK (sample_n >= 0),
    method TEXT NOT NULL,
    calibration_version TEXT,
    UNIQUE (analysis_id, label),
    CHECK (probability_low IS NULL OR probability_low BETWEEN 0.0 AND 1.0),
    CHECK (probability_mid IS NULL OR probability_mid BETWEEN 0.0 AND 1.0),
    CHECK (probability_high IS NULL OR probability_high BETWEEN 0.0 AND 1.0),
    CHECK (
        probability_low IS NULL OR probability_mid IS NULL OR probability_high IS NULL
        OR (probability_low <= probability_mid AND probability_mid <= probability_high)
    ),
    CHECK (return_p10 IS NULL OR return_p50 IS NULL OR return_p90 IS NULL
        OR (return_p10 <= return_p50 AND return_p50 <= return_p90))
);

CREATE TABLE IF NOT EXISTS portfolio_sets (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    risk_profile TEXT NOT NULL CHECK (risk_profile IN ('conservative', 'balanced', 'aggressive')),
    cash_weight REAL NOT NULL CHECK (cash_weight BETWEEN 0.0 AND 1.0),
    borrowed_weight REAL NOT NULL DEFAULT 0.0 CHECK (borrowed_weight >= 0.0),
    expected_return REAL,
    expected_vol REAL,
    expected_max_drawdown REAL,
    sharpe REAL,
    metric_window TEXT NOT NULL,
    UNIQUE (run_id, risk_profile)
);

CREATE TABLE IF NOT EXISTS portfolio_members (
    portfolio_id TEXT NOT NULL REFERENCES portfolio_sets(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    weight REAL NOT NULL CHECK (weight > 0.0 AND weight <= 1.0),
    rank INTEGER NOT NULL CHECK (rank > 0),
    reason_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (portfolio_id, symbol),
    UNIQUE (portfolio_id, rank)
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    symbol TEXT,
    evidence_type TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    url TEXT,
    content_hash TEXT NOT NULL,
    summary TEXT,
    UNIQUE (run_id, content_hash)
);

CREATE TABLE IF NOT EXISTS outcomes (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES stock_analyses(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    realized_return REAL,
    max_drawdown REAL,
    max_runup REAL,
    relative_return REAL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'unavailable')),
    UNIQUE (analysis_id, observed_at)
);

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    shares REAL NOT NULL CHECK (shares >= 0.0),
    cost_price REAL NOT NULL CHECK (cost_price >= 0.0),
    as_of TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_as_of ON runs(as_of);
CREATE INDEX IF NOT EXISTS idx_snapshots_run ON data_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_analyses_symbol_horizon
    ON stock_analyses(symbol, horizon_sessions, run_id);
CREATE INDEX IF NOT EXISTS idx_scenarios_analysis ON scenarios(analysis_id);
CREATE INDEX IF NOT EXISTS idx_evidence_run_symbol ON evidence(run_id, symbol);
CREATE INDEX IF NOT EXISTS idx_outcomes_analysis ON outcomes(analysis_id);

-- A prediction is an audit record. Corrections create a new run; they never
-- rewrite what the program said earlier. Outcomes and positions are excluded
-- because they are intentionally updated as new observations arrive.
CREATE TRIGGER IF NOT EXISTS immutable_runs_update
BEFORE UPDATE ON runs BEGIN
    SELECT RAISE(ABORT, 'runs are immutable; create a new run');
END;
CREATE TRIGGER IF NOT EXISTS immutable_runs_delete
BEFORE DELETE ON runs BEGIN
    SELECT RAISE(ABORT, 'runs are immutable; retain the audit trail');
END;
CREATE TRIGGER IF NOT EXISTS immutable_snapshots_update
BEFORE UPDATE ON data_snapshots BEGIN
    SELECT RAISE(ABORT, 'data snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_snapshots_delete
BEFORE DELETE ON data_snapshots BEGIN
    SELECT RAISE(ABORT, 'data snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_analyses_update
BEFORE UPDATE ON stock_analyses BEGIN
    SELECT RAISE(ABORT, 'stock analyses are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_analyses_delete
BEFORE DELETE ON stock_analyses BEGIN
    SELECT RAISE(ABORT, 'stock analyses are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_scenarios_update
BEFORE UPDATE ON scenarios BEGIN
    SELECT RAISE(ABORT, 'scenarios are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_scenarios_delete
BEFORE DELETE ON scenarios BEGIN
    SELECT RAISE(ABORT, 'scenarios are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_portfolios_update
BEFORE UPDATE ON portfolio_sets BEGIN
    SELECT RAISE(ABORT, 'portfolio sets are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_portfolios_delete
BEFORE DELETE ON portfolio_sets BEGIN
    SELECT RAISE(ABORT, 'portfolio sets are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_members_update
BEFORE UPDATE ON portfolio_members BEGIN
    SELECT RAISE(ABORT, 'portfolio members are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_members_delete
BEFORE DELETE ON portfolio_members BEGIN
    SELECT RAISE(ABORT, 'portfolio members are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_evidence_update
BEFORE UPDATE ON evidence BEGIN
    SELECT RAISE(ABORT, 'evidence records are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_evidence_delete
BEFORE DELETE ON evidence BEGIN
    SELECT RAISE(ABORT, 'evidence records are immutable');
END;

INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
