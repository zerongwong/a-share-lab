"""Local outbox: archive before sending; acceptance is not phone delivery."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime


def put_alert(repository, *, incident, position_key, portfolio, kind, now, confirmed, payload):
    key = hashlib.sha256(incident.encode()).hexdigest()
    with repository.connection() as connection:
        connection.execute(
            """INSERT INTO intraday_stop_alerts
               (incident_key, position_key, portfolio_id, holding_version, kind,
                observed_on, observed_at, confirmed, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            (
                key,
                position_key,
                portfolio.id,
                portfolio.version,
                kind,
                now.date().isoformat(),
                now.astimezone(UTC).isoformat(),
                int(confirmed),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
    return key


def pending_alerts(repository, *, position_keys, portfolio_id, now):
    with repository.connection() as connection:
        rows = connection.execute(
            "SELECT * FROM intraday_stop_alerts WHERE delivery_status != 'accepted' ORDER BY observed_at"
        ).fetchall()
    result = []
    for raw in rows:
        row = dict(raw)
        if row["position_key"] is not None:
            if row["position_key"] not in position_keys:
                continue
        elif row["portfolio_id"] != portfolio_id or row["observed_on"] != now.date().isoformat():
            continue
        if row["kind"] != "cost_exit" and row["observed_on"] != now.date().isoformat():
            continue
        # Never submit an alert whose observation is in the future.
        if datetime.fromisoformat(row["observed_at"]) > now:
            continue
        if row["attempts"] >= 3 and row["kind"] != "cost_exit":
            continue
        if row["attempted_at"] and (
            now - datetime.fromisoformat(row["attempted_at"])
        ).total_seconds() < (180 if row["attempts"] < 3 else 3600):
            continue
        row["payload"] = json.loads(row["payload_json"])
        result.append(row)
    return sorted(result, key=lambda r: (r["kind"] != "cost_exit", r["observed_at"]))[:10]


def mark_attempt(repository, key, now):
    with repository.connection() as connection:
        connection.execute(
            "UPDATE intraday_stop_alerts SET delivery_status='attempted', attempts=attempts+1, attempted_at=? WHERE incident_key=?",
            (now.astimezone(UTC).isoformat(), key),
        )


def mark_delivery(repository, key, accepted):
    with repository.connection() as connection:
        connection.execute(
            "UPDATE intraday_stop_alerts SET delivery_status=? WHERE incident_key=?",
            ("accepted" if accepted else "failed", key),
        )


def confirmed_cost_touch(repository, position_key, *, cutoff: date, known_at: datetime):
    with repository.connection() as connection:
        row = connection.execute(
            """SELECT MIN(observed_on) AS first_touch FROM intraday_stop_alerts
               WHERE position_key=? AND kind='cost_exit' AND confirmed=1
               AND observed_on<=? AND observed_at<=?""",
            (position_key, cutoff.isoformat(), known_at.astimezone(UTC).isoformat()),
        ).fetchone()
    return None if row["first_touch"] is None else date.fromisoformat(row["first_touch"])
