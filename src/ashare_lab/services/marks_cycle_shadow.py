"""Local append-only evidence and daily shadow comparison; no production writes."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from ashare_lab.analytics.marks_cycle import DIMENSIONS, METHOD, assess_marks_cycle


def _schema(connection):
    for table in ("marks_cycle_evidence", "marks_cycle_runs"):
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        for op in ("UPDATE", "DELETE"):
            connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT,'immutable cycle archive'); END"
            )
        connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_replace BEFORE INSERT ON {table} WHEN EXISTS (SELECT 1 FROM {table} WHERE id=NEW.id) BEGIN SELECT RAISE(ABORT,'immutable cycle archive'); END"
        )


def _append(connection, table, document):
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
    identifier = hashlib.sha256(payload.encode()).hexdigest()
    if connection.execute(f"SELECT 1 FROM {table} WHERE id=?", (identifier,)).fetchone() is None:
        connection.execute(f"INSERT INTO {table} VALUES (?,?)", (identifier, payload))
    return {"id": identifier, **document}


def import_evidence(repository, observations, *, now=None):
    """Use independently reviewed metrics, not narrative generated scores.

    recorded_at is always the current import time, never user-backdated. Each
    item names its metric definition/reference sample and original publisher.
    Accepting a file validates schema/provenance shape, not truth of its content.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("aware import time required")
    clean = []
    keys = {
        "dimension",
        "value",
        "observed_through",
        "published_at",
        "retrieved_at",
        "publisher",
        "source_url",
        "metric_definition",
        "reference_window",
        "sample_size",
    }
    for obs in observations:
        if set(obs) != keys or obs["dimension"] not in DIMENSIONS:
            raise ValueError("cycle evidence fields/dimension invalid")
        value = obs["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError("cycle metric must be numeric in [0,100]")
        if type(obs["sample_size"]) is not int or obs["sample_size"] < 20:
            raise ValueError("document at least 20 independent metric/reference observations")
        for key in ("publisher", "source_url", "metric_definition", "reference_window"):
            if not isinstance(obs[key], str) or not obs[key].strip() or len(obs[key]) > 2000:
                raise ValueError("nonempty bounded provenance required")
        url = urlparse(obs["source_url"])
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.fragment
        ):
            raise ValueError("public HTTPS evidence URL required")
        published, retrieved = [
            datetime.fromisoformat(obs[key]) for key in ("published_at", "retrieved_at")
        ]
        observed = date.fromisoformat(obs["observed_through"])
        if (
            published.tzinfo is None
            or retrieved.tzinfo is None
            or not published <= retrieved <= now
            or observed > published.astimezone(ZoneInfo("Asia/Shanghai")).date()
        ):
            raise ValueError("invalid evidence knowledge chronology")
        clean.append({**obs, "recorded_at": now.isoformat()})
    with repository.connection() as connection:
        _schema(connection)
        return [_append(connection, "marks_cycle_evidence", obs) for obs in clean]


def run_marks_cycle_shadow(
    repository, *, price_cutoff: date, known_at: datetime, incumbent_cap: float
):
    with repository.connection() as connection:
        _schema(connection)
        evidence = [
            {"id": row[0], **json.loads(row[1])}
            for row in connection.execute("SELECT id,payload FROM marks_cycle_evidence")
        ]
        previous_runs = [
            json.loads(row[0]) for row in connection.execute("SELECT payload FROM marks_cycle_runs")
        ]
        previous_runs = [
            r
            for r in previous_runs
            if r["method_version"] == METHOD
            and datetime.fromisoformat(r["known_at"]) < known_at
            and r.get("weekly_state_date")
        ]
        previous = max(
            previous_runs, key=lambda r: datetime.fromisoformat(r["known_at"]), default=None
        )
        result = assess_marks_cycle(
            evidence,
            price_cutoff=price_cutoff,
            known_at=known_at,
            incumbent_cap=incumbent_cap,
            previous=previous,
        )
        return _append(connection, "marks_cycle_runs", result)
