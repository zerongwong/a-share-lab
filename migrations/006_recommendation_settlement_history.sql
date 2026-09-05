-- Preserve earlier outcome observations when newly verified evidence arrives.
CREATE TABLE IF NOT EXISTS recommendation_settlement_history (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES recommendation_batches(id) ON DELETE CASCADE,
    evaluated_at TEXT NOT NULL,
    method_version TEXT NOT NULL,
    snapshot_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_settlement_history_batch
    ON recommendation_settlement_history(batch_id, evaluated_at);
CREATE TRIGGER IF NOT EXISTS immutable_settlement_history_update
BEFORE UPDATE ON recommendation_settlement_history BEGIN
    SELECT RAISE(ABORT, 'settlement history is immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_settlement_history_delete
BEFORE DELETE ON recommendation_settlement_history BEGIN
    SELECT RAISE(ABORT, 'settlement history is immutable');
END;
INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (6, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
