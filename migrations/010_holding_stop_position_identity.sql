PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

-- A protective stop belongs to one immutable position identity.  The former
-- UNIQUE(symbol, entry_date) rule incorrectly joined a later holding revision
-- to an older stop when the same stock was bought again on the same date.
-- SQLite cannot drop a table constraint in place, so rebuild the small mutable
-- stop-state table atomically and preserve every historical row verbatim.
BEGIN IMMEDIATE;

CREATE TABLE holding_protective_stops_v2 (
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
    updated_at TEXT NOT NULL
);

INSERT INTO holding_protective_stops_v2 (
    position_key,
    symbol,
    entry_date,
    effective_stop,
    candidate_stop,
    previous_stop,
    data_cutoff,
    source_timeframe,
    evidence_date,
    holding_version,
    method_version,
    details_json,
    updated_at
)
SELECT
    position_key,
    symbol,
    entry_date,
    effective_stop,
    candidate_stop,
    previous_stop,
    data_cutoff,
    source_timeframe,
    evidence_date,
    holding_version,
    method_version,
    details_json,
    updated_at
FROM holding_protective_stops;

DROP TABLE holding_protective_stops;
ALTER TABLE holding_protective_stops_v2 RENAME TO holding_protective_stops;

CREATE INDEX idx_holding_stops_symbol_entry
    ON holding_protective_stops(symbol, entry_date);

CREATE TRIGGER holding_stop_never_moves_down
BEFORE UPDATE OF effective_stop ON holding_protective_stops
WHEN NEW.effective_stop < OLD.effective_stop BEGIN
    SELECT RAISE(ABORT, 'effective holding protection stop cannot move down');
END;

CREATE TRIGGER holding_stop_cutoff_never_moves_back
BEFORE UPDATE OF data_cutoff ON holding_protective_stops
WHEN NEW.data_cutoff < OLD.data_cutoff BEGIN
    SELECT RAISE(ABORT, 'effective holding protection cutoff cannot move backwards');
END;

INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (10, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

COMMIT;
