-- Independent observations/outbox: never orders or holding mutations.
CREATE TABLE IF NOT EXISTS intraday_stop_alerts (
    incident_key TEXT PRIMARY KEY,
    position_key TEXT,
    portfolio_id TEXT NOT NULL,
    holding_version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    observed_on TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
    payload_json TEXT NOT NULL,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    attempted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_intraday_stop_position
ON intraday_stop_alerts(position_key, kind, observed_on);
