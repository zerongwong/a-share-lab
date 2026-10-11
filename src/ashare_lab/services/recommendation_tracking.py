"""Independent, immutable recommendation lifecycle research (never actual fills).

Daily OHLC simulation, not tick replay: published plan -> complete daily signal
-> next session open within the archived buy cap. A pullback must touch the
archived range and close back above its upper edge with close >= open. An
untriggered daily plan expires; an entered position has NO fixed maturity.

The 8% cost guard uses the simulated fill, not the recommendation reference.
Daily structural failure exits next tradable open without waiting for a rebound.
Single-price/zero-volume bars cannot establish executable liquidity. Missing
bars or corporate-action evidence stop accounting, never manufacture a fill.
All results are gross hypothetical PRICE returns, not account or total returns.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ashare_lab.services.continuous_strategy_journal import _savepoint

TRACKING_START = date(2026, 10, 8)
TRACKING_METHOD = "recommendation-lifecycle-v1"
SHANGHAI = ZoneInfo("Asia/Shanghai")
_ACTIVE = {
    "awaiting_confirmation", "pending_entry", "open", "pending_exit",
    "evidence_missing", "corporate_action_review_required",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _aware(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("tracking timestamp must be timezone-aware")
    return value.isoformat()


def ensure_tracking_schema(connection: sqlite3.Connection) -> None:
    """Only add two research tables; never change holdings or legacy cohorts."""
    with _savepoint(connection):
        connection.execute("""CREATE TABLE IF NOT EXISTS tracking_reports (
            report_id TEXT PRIMARY KEY NOT NULL, plan_date TEXT NOT NULL,
            payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS tracking_runs (
            run_hash TEXT PRIMARY KEY NOT NULL, through_date TEXT NOT NULL,
            created_at TEXT NOT NULL, input_hash TEXT NOT NULL,
            result_json TEXT NOT NULL, result_hash TEXT NOT NULL)""")
        for table, key in (("tracking_reports", "report_id"), ("tracking_runs", "run_hash")):
            for op in ("UPDATE", "DELETE"):
                connection.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()}
                    BEFORE {op} ON {table}
                    BEGIN SELECT RAISE(ABORT, 'tracking archive is immutable'); END""")
            connection.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_replace
                BEFORE INSERT ON {table}
                WHEN EXISTS (SELECT 1 FROM {table} WHERE {key}=NEW.{key})
                BEGIN SELECT RAISE(ABORT, 'tracking archive is immutable'); END""")


def ingest_report(connection: sqlite3.Connection, *, report_id: str, plan_date: date,
                  published_at: datetime, cutoff: date, method_version: str,
                  entries: list[dict]) -> bool:
    """Archive an accepted original report, including zero-entry reports.

    Caller must verify original report, immutable decision hash and provider
    acceptance. Observation candidates never qualify. Unknown rules remain
    unknown; display strings are not reverse-engineered into execution rules.
    """
    if plan_date < TRACKING_START:
        return False
    if not report_id or not method_version or cutoff >= plan_date:
        raise ValueError("invalid original recommendation chronology")
    unique = {}
    for entry in entries:
        if entry.get("entry_qualified") is not True:
            continue
        weight = _number(entry.get("account_weight"))
        if weight is None or weight <= 0:
            continue
        symbol = str(entry.get("symbol", ""))
        if len(symbol) != 6 or not symbol.isdecimal():
            raise ValueError("invalid public stock code")
        copied = json.loads(_json(entry))
        if symbol in unique and unique[symbol] != copied:
            raise ValueError("conflicting same-report recommendations")
        unique[symbol] = copied
    payload = {"report_id": report_id, "plan_date": plan_date.isoformat(),
               "published_at": _aware(published_at), "cutoff": cutoff.isoformat(),
               "method_version": method_version,
               "entries": [unique[s] for s in sorted(unique)]}
    encoded = _json(payload)
    with _savepoint(connection):
        ensure_tracking_schema(connection)
        old = connection.execute(
            "SELECT payload_json,payload_hash FROM tracking_reports WHERE report_id=?",
            (report_id,),
        ).fetchone()
        if old:
            if _hash(old[0]) != old[1] or old[0] != encoded:
                raise ValueError("original tracking report changed")
            return False
        connection.execute("INSERT INTO tracking_reports VALUES (?,?,?,?)",
                           (report_id, plan_date.isoformat(), encoded, _hash(encoded)))
    return True


def _reports(connection: sqlite3.Connection) -> list[dict]:
    result = []
    for payload, digest in connection.execute(
        "SELECT payload_json,payload_hash FROM tracking_reports ORDER BY plan_date,report_id"
    ):
        if _hash(payload) != digest:
            raise ValueError("tracking report integrity failure")
        result.append(json.loads(payload))
    return sorted(result, key=lambda r: (
        r["plan_date"], datetime.fromisoformat(r["published_at"]).astimezone(UTC), r["report_id"]
    ))


def _new_episode(report: dict, entry: dict) -> dict:
    rule = entry.get("entry_rule") or {}
    stop = _number(rule.get("structural_invalidation_price"))
    if stop is None:
        stop = _number(entry.get("structural_invalidation_price"))
    return {
        "episode_id": _hash(report["report_id"] + ":" + entry["symbol"])[:24],
        "symbol": entry["symbol"], "name": entry.get("name", entry["symbol"]),
        "report_ids": [report["report_id"]], "plan_date": report["plan_date"],
        "published_at": report["published_at"], "data_cutoff": report["cutoff"],
        "recommendation_method": report["method_version"], "entry_rule": rule,
        "planned_price_low": rule.get("price_low"),
        "planned_price_high": rule.get("price_high"),
        "planned_trigger_price": rule.get("trigger_price"),
        "recommendation_reference_price": rule.get("entry_reference_price"),
        "status": "awaiting_confirmation", "reason": "awaiting_plan_session",
        "entry_date": None, "entry_price": None, "exit_date": None, "exit_price": None,
        "profit_per_share": None, "return_fraction": None,
        "daily_average_return": None, "daily_compound_return": None,
        "trading_days": None, "calendar_days": None,
        "unrealized_return": None, "unrealized_profit_per_share": None,
        "stop_price": stop, "initial_structural_stop": stop,
        "cost_stop_price": None, "events": [],
        "last_valued_date": None, "last_price": None,
        "accounting_basis": "hypothetical_gross_price_return_no_fees_or_dividends",
    }


def _event(episode: dict, day: date, kind: str, **detail: Any) -> None:
    episode["events"].append({"date": day.isoformat(), "kind": kind, **detail})


def _valid_bar(bar: dict | None) -> bool:
    if not bar:
        return False
    nums = [_number(bar.get(k)) for k in ("open", "high", "low", "close")]
    if any(n is None or n <= 0 for n in nums):
        return False
    op, high, low, close = nums
    return low <= min(op, close) <= max(op, close) <= high


def _liquid(bar: dict) -> bool:
    volume = _number(bar.get("volume_shares"))
    return volume is not None and volume > 0 and float(bar["high"]) > float(bar["low"])


def _clear(evidence: dict, start: date, end: date) -> bool:
    try:
        return (
            evidence.get("complete") is True
            and evidence.get("status") in {"clear", "detected"}
            and date.fromisoformat(evidence["coverage_from"]) <= start
            and date.fromisoformat(evidence["coverage_through"]) >= end
            and not any(start <= date.fromisoformat(d) <= end
                        for d in evidence.get("event_dates", []))
            # Detected without dates cannot attest absence in an interval.
            and (evidence.get("status") != "detected" or bool(evidence.get("event_dates")))
        )
    except (TypeError, ValueError, KeyError):
        return False


def _valid_rule(rule: dict, stop: float | None) -> bool:
    cap = _number(rule.get("maximum_entry_price"))
    if rule.get("schema_version") != "recommendation-entry-rule-v1":
        return False
    if stop is None or stop <= 0 or cap is None or cap <= stop:
        return False
    kind = rule.get("kind")
    if kind == "healthy_pullback_range":
        lo, hi = _number(rule.get("price_low")), _number(rule.get("price_high"))
        return lo is not None and hi is not None and 0 < lo <= hi <= cap
    trigger = _number(rule.get("trigger_price"))
    if trigger is None or not 0 < trigger <= cap:
        return False
    if kind == "reclaim_close_confirmation":
        return True
    return (kind == "volume_breakout_close_confirmation"
            and rule.get("confirmation_activity_metric") in {"volume_shares", "amount_cny"}
            and (_number(rule.get("confirmation_activity_min")) or 0) > 0)


def _confirm(rule: dict, bar: dict, stop: float) -> bool:
    close = float(bar["close"])
    if close <= stop or close > float(rule["maximum_entry_price"]) or not _liquid(bar):
        return False
    if rule["kind"] == "healthy_pullback_range":
        # Contact alone isn't stabilization. No assumption of filling at daily low.
        return (float(bar["low"]) <= float(rule["price_high"])
                and float(bar["high"]) >= float(rule["price_low"])
                and close >= float(rule["price_high"]) and close >= float(bar["open"]))
    if close < float(rule["trigger_price"]):
        return False
    if rule["kind"] == "volume_breakout_close_confirmation":
        observed = _number(bar.get(rule["confirmation_activity_metric"]))
        return observed is not None and observed >= float(rule["confirmation_activity_min"])
    return True


def _close(episode: dict, day: date, price: float, reason: str,
           sessions: list[date]) -> None:
    entered = date.fromisoformat(episode["entry_date"])
    n = sum(entered <= d <= day for d in sessions)
    profit = price - episode["entry_price"]
    ret = profit / episode["entry_price"]
    episode.update(status="closed", reason=reason, exit_date=day.isoformat(),
                   exit_price=price, profit_per_share=profit, return_fraction=ret,
                   trading_days=n, calendar_days=(day - entered).days + 1,
                   daily_average_return=ret / n, daily_compound_return=(1 + ret) ** (1 / n) - 1,
                   unrealized_return=None, unrealized_profit_per_share=None,
                   last_valued_date=day.isoformat(), last_price=price)
    _event(episode, day, "simulated_exit", price=price, reason=reason)


def _ratchet(episode: dict, day: date, history: dict[date, dict]) -> None:
    """Use existing weekly swing/ATR rule; confirmed now, effective next session.

    Daily closes determine exit; completed weekly lows locate support. Inputs
    are truncated before constructing bars or pivots to prevent future leakage.
    Initial archived line remains in force while insufficient new history exists.
    """
    import pandas as pd

    from ashare_lab.analytics.indicators import atr
    from ashare_lab.analytics.multi_timeframe import build_completed_timeframes
    from ashare_lab.analytics.trend import confirmed_swings
    from ashare_lab.services.review_active_holdings import ATR_BUFFER_MULTIPLE

    records = [{**bar, "trade_date": d} for d, bar in sorted(history.items())
               if d <= day and _valid_bar(bar)]
    if len(records) < 15:
        return
    frame = pd.DataFrame(records)
    completed = build_completed_timeframes(frame, as_of=day)
    weekly = completed.weekly
    points = [p for p in confirmed_swings(weekly, left=2, right=2)
              if p.kind == "low" and p.trade_date.date() > date.fromisoformat(episode["entry_date"])]
    if not points:
        return
    point = points[-1]
    atr_rows = frame[pd.to_datetime(frame["trade_date"]).dt.date <= point.confirmed_at.date()]
    volatility = float(atr(atr_rows, 14).iloc[-1])
    if not math.isfinite(volatility) or volatility <= 0:
        return
    candidate = math.floor((float(point.price) - ATR_BUFFER_MULTIPLE * volatility) * 100) / 100
    if candidate > episode["stop_price"]:
        episode["stop_price"] = candidate
        _event(episode, day, "raise_structural_stop_next_session", price=candidate,
               support_date=point.trade_date.date().isoformat(),
               confirmed_at=point.confirmed_at.date().isoformat())


def _step(episode: dict, day: date, history: dict[date, dict], evidence: dict,
          sessions: list[date]) -> None:
    if episode["status"] in {"closed", "unfilled", "evidence_missing",
                             "corporate_action_review_required", "rule_missing"}:
        return
    rule = episode["entry_rule"]
    if not _valid_rule(rule, episode["initial_structural_stop"]):
        episode.update(status="rule_missing", reason="structured_original_rule_unavailable")
        return
    start = date.fromisoformat(episode["data_cutoff"]) + timedelta(days=1)
    if not _clear(evidence, start, day):
        episode.update(status="corporate_action_review_required",
                       reason="corporate_actions_unknown_or_price_basis_changed")
        _event(episode, day, "accounting_blocked", reason=episode["reason"])
        return
    bar = history.get(day)
    if not _valid_bar(bar):
        episode.update(status="evidence_missing", reason="missing_or_invalid_session_bar")
        _event(episode, day, "accounting_blocked", reason=episode["reason"])
        return
    if _number(bar.get("volume_shares")) is None or (
        episode["status"] == "awaiting_confirmation"
        and rule["kind"] == "volume_breakout_close_confirmation"
        and _number(bar.get(rule["confirmation_activity_metric"])) is None
    ):
        episode.update(status="evidence_missing", reason="liquidity_or_confirmation_activity_unknown")
        _event(episode, day, "accounting_blocked", reason=episode["reason"])
        return
    op, low, close = (float(bar[k]) for k in ("open", "low", "close"))
    liquid = _liquid(bar)
    if episode["status"] == "awaiting_confirmation":
        published = datetime.fromisoformat(episode["published_at"]).astimezone(SHANGHAI)
        open_time = datetime.combine(day, time(9, 30), tzinfo=SHANGHAI)
        if published >= open_time:
            episode.update(status="unfilled", reason="late_publication_no_retroactive_entry")
        elif _confirm(rule, bar, episode["stop_price"]):
            episode.update(status="pending_entry", reason="daily_confirmation_next_open")
            _event(episode, day, "entry_confirmed_at_close")
        else:
            episode.update(status="unfilled", reason="daily_entry_condition_not_confirmed")
        return
    if episode["status"] == "pending_entry":
        if not liquid or not episode["stop_price"] < op <= float(rule["maximum_entry_price"]):
            episode.update(status="unfilled", reason="next_open_unexecutable_or_outside_price_cap")
            return
        episode.update(status="open", reason="hypothetical_next_open_fill",
                       entry_date=day.isoformat(), entry_price=op, cost_stop_price=op * 0.92)
        _event(episode, day, "simulated_entry", price=op)
    if episode["status"] == "pending_exit" and liquid:
        _close(episode, day, op, episode["reason"], sessions)
        return
    # A stop cannot fill at the ideal threshold after a gap or on a locked board.
    if low <= episode["cost_stop_price"]:
        if day.isoformat() != episode["entry_date"] and liquid:
            _close(episode, day, min(op, episode["cost_stop_price"]), "cost_8pct_stop", sessions)
            return
        episode.update(status="pending_exit", reason="cost_8pct_stop_deferred_t1_or_liquidity")
        _event(episode, day, "cost_stop_latched")
    elif close < episode["stop_price"] and episode["status"] != "pending_exit":
        episode.update(status="pending_exit", reason="daily_structural_break_next_open")
        _event(episode, day, "structural_exit_confirmed_at_close", price=episode["stop_price"])
    episode.update(last_valued_date=day.isoformat(), last_price=close,
                   unrealized_return=close / episode["entry_price"] - 1,
                   unrealized_profit_per_share=close - episode["entry_price"],
                   trading_days=sum(date.fromisoformat(episode["entry_date"]) <= d <= day
                                    for d in sessions),
                   calendar_days=(day - date.fromisoformat(episode["entry_date"])).days + 1)
    if episode["status"] == "open":
        _ratchet(episode, day, history)
        if close < episode["stop_price"]:
            episode.update(status="pending_exit", reason="newly_confirmed_structural_break_next_open")
            _event(episode, day, "structural_exit_confirmed_at_close", price=episode["stop_price"])


def replay_tracking(connection: sqlite3.Connection, *, sessions: list[date],
                    bars: dict[str, dict[date, dict]], through: date,
                    clearances: dict[str, dict], run_at: datetime) -> dict:
    """Deterministic replay + immutable hashed observation, never a live order.

    Runner supplies a verified, gap-free exchange session chain and raw bars.
    Unknown bar/action evidence is a separate status, not a losing/winning trade.
    A repaired input creates a new observation without rewriting earlier evidence.
    """
    _aware(run_at)
    if sessions != sorted(set(sessions)):
        raise ValueError("sessions must be a unique ascending verified exchange calendar")
    sessions = [d for d in sessions if d <= through]
    ensure_tracking_schema(connection)
    reports = _reports(connection)
    usable = [r for r in reports if date.fromisoformat(r["plan_date"]) <= through]
    by_day: dict[date, list[tuple[dict, dict]]] = {}
    for report in usable:
        for entry in report["entries"]:
            by_day.setdefault(date.fromisoformat(report["plan_date"]), []).append((report, entry))
    episodes: list[dict] = []
    active: dict[str, dict] = {}
    for day in sorted(set(sessions) | set(by_day)):
        # Resolve a previously scheduled next-open fill/exit before linking
        # today's recommendation. Otherwise an expired old buy cap or an exit
        # could swallow a valid new plan into a now-terminal episode.
        processed = set()
        if day in sessions:
            for episode in active.values():
                if date.fromisoformat(episode["plan_date"]) < day:
                    _step(episode, day, bars.get(episode["symbol"], {}),
                          clearances.get(episode["symbol"], {}), sessions)
                    processed.add(episode["episode_id"])
        for report, entry in by_day.get(day, []):
            old = active.get(entry["symbol"])
            blocked_before_entry = old and old["entry_price"] is None and old["status"] in {
                "evidence_missing", "corporate_action_review_required",
            }
            if old and old["status"] in _ACTIVE and not blocked_before_entry:
                if report["report_id"] not in old["report_ids"]:
                    old["report_ids"].append(report["report_id"])
                    _event(old, day, "recommended_again_not_new_trade", report_id=report["report_id"])
                continue
            new = _new_episode(report, entry)
            episodes.append(new)
            active[entry["symbol"]] = new
            if day not in sessions:
                new.update(status="evidence_missing", reason="plan_session_calendar_unverified")
        if day in sessions:
            for episode in active.values():
                if (date.fromisoformat(episode["plan_date"]) <= day
                        and episode["episode_id"] not in processed):
                    _step(episode, day, bars.get(episode["symbol"], {}),
                          clearances.get(episode["symbol"], {}), sessions)
    counts = dict(Counter(e["status"] for e in episodes))
    closed = [e for e in episodes if e["status"] == "closed"]
    returns = [e["return_fraction"] for e in closed]
    diagnostics = {
        "status_counts": counts, "completed_trades": len(closed),
        "winning_trades": sum(r > 0 for r in returns),
        "losing_trades": sum(r < 0 for r in returns),
        "closed_trade_mean_return": sum(returns) / len(returns) if returns else None,
        "closed_trade_mean_is_not_portfolio_return": True,
        "excluded_incomplete_episodes": sum(counts.get(s, 0) for s in (
            "evidence_missing", "corporate_action_review_required", "rule_missing")),
        "review_status": "insufficient_completed_samples" if len(closed) < 30 else "review_required",
        "rule_changes_require_user_approval": True,
        "rules_changed_automatically": False,
        "research_flags": (["inspect_gap_or_liquidity_losses"]
                           if any(r < -0.080001 for r in returns) else []),
    }
    relevant = {s: {d.isoformat(): bar for d, bar in sorted(history.items()) if d <= through}
                for s, history in sorted(bars.items()) if s in active}
    input_hash = _hash(_json({"reports": usable, "sessions": [d.isoformat() for d in sessions],
                             "bars": relevant, "clearances": clearances}))
    result = {"method_version": TRACKING_METHOD, "through": through.isoformat(),
              "reports": len(usable), "mentions": sum(len(r["entries"]) for r in usable),
              "zero_entry_reports": sum(not r["entries"] for r in usable),
              "episodes": episodes, "diagnostics": diagnostics,
              "execution_model": "daily_confirmation_next_open_t1_gross_price_simulation",
              "orders_enabled": False, "holdings_modified": False,
              "input_hash": input_hash}
    encoded = _json(result)
    result_hash = _hash(encoded)
    run_hash = _hash(TRACKING_METHOD + through.isoformat() + input_hash)
    with _savepoint(connection):
        old = connection.execute("SELECT result_hash FROM tracking_runs WHERE run_hash=?",
                                 (run_hash,)).fetchone()
        if old and old[0] != result_hash:
            raise ValueError("same inputs produced different tracking results; bump method version")
        if not old:
            connection.execute("INSERT INTO tracking_runs VALUES (?,?,?,?,?,?)",
                               (run_hash, through.isoformat(), run_at.isoformat(), input_hash,
                                encoded, result_hash))
    return result


def tracking_summary(connection: sqlite3.Connection) -> dict:
    ensure_tracking_schema(connection)
    row = connection.execute("""SELECT result_json,result_hash FROM tracking_runs
        ORDER BY through_date DESC,created_at DESC,rowid DESC LIMIT 1""").fetchone()
    if row:
        if _hash(row[0]) != row[1]:
            raise ValueError("tracking result integrity failure")
        return json.loads(row[0])
    reports = _reports(connection)
    return {"status": "awaiting_first_replay", "reports": len(reports),
            "mentions": sum(len(r["entries"]) for r in reports), "episodes": [],
            "diagnostics": {"rules_changed_automatically": False}}
