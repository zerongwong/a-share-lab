-- Sanitised, append-only evidence for consent-gated holding company-action checks.
--
-- Version 9 deliberately uses a new table name.  An early, unreleased version
-- 8 may already have been created by a local background worker with a narrower
-- schema; retaining it avoids destructive migration while all production reads
-- use this complete, method-bound attempt ledger.
CREATE TABLE IF NOT EXISTS company_action_evidence_attempts (
    evidence_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    position_key TEXT NOT NULL,
    holding_version INTEGER NOT NULL CHECK (holding_version > 0),
    symbol TEXT NOT NULL CHECK (
        length(symbol) = 6
        AND symbol NOT GLOB '*[^0-9]*'
    ),
    as_of TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('intraday', 'eod')),
    attempt INTEGER NOT NULL CHECK (attempt BETWEEN 1 AND 3),
    status TEXT NOT NULL CHECK (status IN ('clear', 'detected', 'unknown')),
    coverage_from TEXT,
    coverage_through TEXT,
    knowledge_time TEXT NOT NULL,
    events_json TEXT NOT NULL DEFAULT '[]',
    provider_response_hash TEXT,
    stream_receipts_json TEXT NOT NULL DEFAULT '[]',
    evidence_hash TEXT NOT NULL CHECK (length(evidence_hash) = 64),
    reason_code TEXT NOT NULL,
    provider_method_version TEXT NOT NULL,
    local_method_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (
        portfolio_id,
        position_key,
        holding_version,
        as_of,
        phase,
        provider_method_version,
        local_method_version,
        attempt
    ),
    CHECK (
        (coverage_from IS NULL AND coverage_through IS NULL)
        OR
        (coverage_from IS NOT NULL AND coverage_through IS NOT NULL
         AND coverage_from <= coverage_through)
    )
);

CREATE INDEX IF NOT EXISTS idx_company_action_attempts_cache
    ON company_action_evidence_attempts(
        portfolio_id,
        position_key,
        holding_version,
        as_of,
        phase,
        provider_method_version,
        local_method_version,
        attempt DESC
    );

CREATE INDEX IF NOT EXISTS idx_company_action_attempts_symbol_knowledge
    ON company_action_evidence_attempts(symbol, knowledge_time);

CREATE TRIGGER IF NOT EXISTS immutable_company_action_attempts_update
BEFORE UPDATE ON company_action_evidence_attempts BEGIN
    SELECT RAISE(ABORT, 'company-action evidence is append-only');
END;

CREATE TRIGGER IF NOT EXISTS immutable_company_action_attempts_delete
BEFORE DELETE ON company_action_evidence_attempts BEGIN
    SELECT RAISE(ABORT, 'company-action evidence is append-only');
END;

-- BEFORE INSERT runs even for INSERT OR REPLACE.  This closes SQLite's default
-- recursive-trigger loophole and prevents replacement of an archived attempt.
CREATE TRIGGER IF NOT EXISTS immutable_company_action_attempts_replace
BEFORE INSERT ON company_action_evidence_attempts
WHEN EXISTS (
    SELECT 1 FROM company_action_evidence_attempts AS existing
    WHERE existing.evidence_id = NEW.evidence_id
       OR (
            existing.portfolio_id = NEW.portfolio_id
        AND existing.position_key = NEW.position_key
        AND existing.holding_version = NEW.holding_version
        AND existing.as_of = NEW.as_of
        AND existing.phase = NEW.phase
        AND existing.provider_method_version = NEW.provider_method_version
        AND existing.local_method_version = NEW.local_method_version
        AND existing.attempt = NEW.attempt
       )
)
BEGIN
    SELECT RAISE(ABORT, 'company-action evidence is append-only');
END;

INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES (9, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
