PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

-- Each explicit user statement creates a new immutable snapshot.  The newest
-- version remains current until another explicit snapshot (including an empty
-- one) is recorded; daily research reviews never rewrite it.
CREATE TABLE IF NOT EXISTS holding_portfolio_revisions (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL UNIQUE CHECK (version > 0),
    holding_weeks INTEGER NOT NULL CHECK (
        holding_weeks IN (1, 2, 4, 13, 26, 52)
    ),
    effective_at TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'cleared')),
    method_version TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS holding_positions (
    id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL
        REFERENCES holding_portfolio_revisions(id) ON DELETE CASCADE,
    position_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    cost_price REAL CHECK (cost_price IS NULL OR cost_price > 0.0),
    stock_sleeve_weight REAL NOT NULL CHECK (
        stock_sleeve_weight > 0.0 AND stock_sleeve_weight <= 1.0
    ),
    account_weight REAL CHECK (
        account_weight IS NULL OR
        (account_weight > 0.0 AND account_weight <= 1.0)
    ),
    status TEXT NOT NULL CHECK (status IN ('active', 'exited')),
    source TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (revision_id, symbol),
    UNIQUE (revision_id, position_key)
);

-- This is the only mutable part of the holding model.  It deliberately keeps
-- the highest confirmed protection line for the same symbol/entry date.
CREATE TABLE IF NOT EXISTS holding_protective_stops (
    position_key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    effective_stop REAL NOT NULL CHECK (effective_stop > 0.0),
    candidate_stop REAL NOT NULL CHECK (candidate_stop > 0.0),
    previous_stop REAL CHECK (previous_stop IS NULL OR previous_stop > 0.0),
    data_cutoff TEXT NOT NULL,
    source_timeframe TEXT NOT NULL,
    evidence_date TEXT NOT NULL,
    holding_version INTEGER NOT NULL CHECK (holding_version > 0),
    method_version TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    UNIQUE (symbol, entry_date)
);

-- Daily decisions are append-only evidence.  They are holding-management
-- actions, never candidate ranks and never broker orders.
CREATE TABLE IF NOT EXISTS holding_review_events (
    id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL
        REFERENCES holding_portfolio_revisions(id) ON DELETE RESTRICT,
    position_id TEXT NOT NULL REFERENCES holding_positions(id) ON DELETE RESTRICT,
    position_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    holding_weeks INTEGER NOT NULL CHECK (
        holding_weeks IN (1, 2, 4, 13, 26, 52)
    ),
    holding_version INTEGER NOT NULL CHECK (holding_version > 0),
    reviewed_at TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ready', 'data_not_ready')),
    holding_action TEXT NOT NULL CHECK (
        holding_action IN ('hold', 'tighten', 'reduce', 'exit', 'review')
    ),
    latest_close REAL CHECK (latest_close IS NULL OR latest_close > 0.0),
    candidate_stop REAL CHECK (candidate_stop IS NULL OR candidate_stop > 0.0),
    previous_stop REAL CHECK (previous_stop IS NULL OR previous_stop > 0.0),
    effective_stop REAL CHECK (effective_stop IS NULL OR effective_stop > 0.0),
    close_below_stop INTEGER CHECK (close_below_stop IN (0, 1)),
    source_timeframe TEXT,
    evidence_date TEXT,
    company_action_clear INTEGER CHECK (company_action_clear IN (0, 1)),
    company_action_evidence_id TEXT,
    company_action_evidence_source TEXT,
    company_action_clear_through TEXT,
    decision_layer TEXT NOT NULL CHECK (decision_layer = 'holding_management'),
    candidate_rank_used INTEGER NOT NULL DEFAULT 0 CHECK (candidate_rank_used = 0),
    next_session_only INTEGER NOT NULL DEFAULT 1 CHECK (next_session_only = 1),
    auto_order_allowed INTEGER NOT NULL DEFAULT 0 CHECK (auto_order_allowed = 0),
    reason_json TEXT NOT NULL DEFAULT '[]',
    evidence_hash TEXT NOT NULL,
    method_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (position_key, holding_version, data_cutoff, method_version, evidence_hash)
);

CREATE INDEX IF NOT EXISTS idx_holding_revisions_version
    ON holding_portfolio_revisions(version DESC);
CREATE INDEX IF NOT EXISTS idx_holding_positions_revision_status
    ON holding_positions(revision_id, status, symbol);
CREATE INDEX IF NOT EXISTS idx_holding_reviews_cutoff_action
    ON holding_review_events(data_cutoff, holding_action, symbol);

CREATE TRIGGER IF NOT EXISTS immutable_holding_revisions_update
BEFORE UPDATE ON holding_portfolio_revisions BEGIN
    SELECT RAISE(ABORT, 'holding portfolio revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_holding_revisions_delete
BEFORE DELETE ON holding_portfolio_revisions BEGIN
    SELECT RAISE(ABORT, 'holding portfolio revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_holding_positions_update
BEFORE UPDATE ON holding_positions BEGIN
    SELECT RAISE(ABORT, 'holding positions are immutable snapshots');
END;
CREATE TRIGGER IF NOT EXISTS immutable_holding_positions_delete
BEFORE DELETE ON holding_positions BEGIN
    SELECT RAISE(ABORT, 'holding positions are immutable snapshots');
END;
CREATE TRIGGER IF NOT EXISTS immutable_holding_reviews_update
BEFORE UPDATE ON holding_review_events BEGIN
    SELECT RAISE(ABORT, 'holding review events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_holding_reviews_delete
BEFORE DELETE ON holding_review_events BEGIN
    SELECT RAISE(ABORT, 'holding review events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS holding_stop_never_moves_down
BEFORE UPDATE OF effective_stop ON holding_protective_stops
WHEN NEW.effective_stop < OLD.effective_stop BEGIN
    SELECT RAISE(ABORT, 'effective holding protection stop cannot move down');
END;
CREATE TRIGGER IF NOT EXISTS holding_stop_cutoff_never_moves_back
BEFORE UPDATE OF data_cutoff ON holding_protective_stops
WHEN NEW.data_cutoff < OLD.data_cutoff BEGIN
    SELECT RAISE(ABORT, 'effective holding protection cutoff cannot move backwards');
END;

INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
