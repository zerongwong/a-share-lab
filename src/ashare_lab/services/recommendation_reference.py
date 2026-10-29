"""Parallel open-to-exit-signal-close price reference, never simulated fills.

This user-requested benchmark deliberately does NOT replace lifecycle-v1's
confirmation/next-open execution model. Formal recommendations published before
the plan-day open get an open-price observation; their original conditional
entry is not assumed to have executed. The first existing exit-rule trigger is
marked at that day's close, even if T+1, a gap or a locked board prevents trading.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta

from ashare_lab.services.continuous_strategy_journal import _savepoint
from ashare_lab.services.recommendation_tracking import (
    SHANGHAI,
    _aware,
    _clear,
    _event,
    _hash,
    _json,
    _new_episode,
    _number,
    _ratchet,
    _reports,
    _valid_bar,
    _valid_rule,
    ensure_tracking_schema,
)

REFERENCE_METHOD = "recommendation-open-signal-close-reference-v1"
VERSION_ISOLATED_REFERENCE_METHOD = "recommendation-open-signal-close-reference-v2-method-isolated"
_TERMINAL = {"closed", "excluded_late_publication", "rule_missing"}
_BLOCKED = {"evidence_missing", "corporate_action_review_required"}


def ensure_reference_schema(connection: sqlite3.Connection) -> None:
    """Separate immutable observations; existing v1 archives stay untouched."""
    ensure_tracking_schema(connection)
    with _savepoint(connection):
        connection.execute("""CREATE TABLE IF NOT EXISTS tracking_reference_runs (
            run_hash TEXT PRIMARY KEY NOT NULL, through_date TEXT NOT NULL,
            created_at TEXT NOT NULL, input_hash TEXT NOT NULL,
            result_json TEXT NOT NULL, result_hash TEXT NOT NULL)""")
        for operation in ("UPDATE", "DELETE"):
            connection.execute(f"""CREATE TRIGGER IF NOT EXISTS
                tracking_reference_runs_no_{operation.lower()}
                BEFORE {operation} ON tracking_reference_runs
                BEGIN SELECT RAISE(ABORT, 'reference archive is immutable'); END""")
        connection.execute("""CREATE TRIGGER IF NOT EXISTS tracking_reference_runs_no_replace
            BEFORE INSERT ON tracking_reference_runs
            WHEN EXISTS (SELECT 1 FROM tracking_reference_runs WHERE run_hash=NEW.run_hash)
            BEGIN SELECT RAISE(ABORT, 'reference archive is immutable'); END""")


def _episode(report: dict, entry: dict) -> dict:
    result = _new_episode(report, entry)
    result.update(
        episode_id=_hash(REFERENCE_METHOD + ":" + report["report_id"] + ":" + entry["symbol"])[:24],
        status="awaiting_open_reference", reason="awaiting_verified_plan_session",
        daily_average_profit_per_share=None, exit_trigger_date=None,
        reference_only=True, execution_eligible=False, actual_trade=False,
        entry_condition_assumed=True,
        entry_condition_execution="not_assumed; see separate lifecycle-v1 simulation",
        accounting_basis="open_to_first_exit_signal_close_gross_price_reference_no_fees_or_dividends",
        execution_limitations=[],
    )
    return result


def _mark(episode: dict, day: date, close: float, sessions: list[date]) -> None:
    entered = date.fromisoformat(episode["entry_date"])
    n = sum(entered <= d <= day for d in sessions)
    profit = close - episode["entry_price"]
    ret = profit / episode["entry_price"]
    episode.update(
        trading_days=n, calendar_days=(day - entered).days + 1,
        last_valued_date=day.isoformat(), last_price=close,
        unrealized_return=ret, unrealized_profit_per_share=profit,
        daily_average_return=ret / n, daily_average_profit_per_share=profit / n,
        daily_compound_return=(1 + ret) ** (1 / n) - 1,
    )


def _finish(episode: dict, day: date, close: float, reason: str, sessions: list[date]) -> None:
    _mark(episode, day, close, sessions)
    episode.update(
        status="closed", reason=reason, exit_trigger_date=day.isoformat(),
        exit_date=day.isoformat(), exit_price=close,
        profit_per_share=episode["unrealized_profit_per_share"],
        return_fraction=episode["unrealized_return"],
        unrealized_return=None, unrealized_profit_per_share=None,
    )
    episode["execution_limitations"].append("signal_day_close_is_not_a_guaranteed_exit_fill")
    if episode["entry_date"] == day.isoformat():
        episode["execution_limitations"].append("same_day_exit_not_executable_under_t1")
    _event(episode, day, "exit_signal_close_reference", price=close, reason=reason)


def _step(episode: dict, day: date, history: dict[date, dict], evidence: dict,
          sessions: list[date]) -> None:
    if episode["status"] in _TERMINAL | _BLOCKED:
        return
    if episode["entry_price"] is None:
        published = datetime.fromisoformat(episode["published_at"]).astimezone(SHANGHAI)
        if published >= datetime.combine(day, time(9, 30), tzinfo=SHANGHAI):
            episode.update(status="excluded_late_publication",
                           reason="late_publication_no_retroactive_open_reference")
            return
    if not _valid_rule(episode["entry_rule"], episode["initial_structural_stop"]):
        episode.update(status="rule_missing", reason="structured_original_rule_unavailable")
        return
    start = date.fromisoformat(episode["data_cutoff"]) + timedelta(days=1)
    if not _clear(evidence, start, day):
        episode.update(status="corporate_action_review_required",
                       reason="corporate_actions_unknown_or_price_basis_changed")
        return
    bar = history.get(day)
    if not _valid_bar(bar) or _number(bar.get("volume_shares")) is None:
        episode.update(status="evidence_missing", reason="missing_or_invalid_verified_session_bar")
        return
    if float(bar["volume_shares"]) <= 0:
        # A stale suspension quote is not an official open or close reference.
        episode.update(status="evidence_missing", reason="no_traded_open_or_close_reference")
        return
    op, low, close = (float(bar[k]) for k in ("open", "low", "close"))
    if episode["entry_price"] is None:
        episode.update(status="open", reason="formal_recommendation_open_price_reference",
                       entry_date=day.isoformat(), entry_price=op, cost_stop_price=op * 0.92)
        rule = episode["entry_rule"]
        flags = ["original_conditional_entry_not_verified_by_open_price"]
        if op <= episode["initial_structural_stop"]:
            flags.append("open_at_or_below_original_structure_stop")
        if op > float(rule["maximum_entry_price"]):
            flags.append("open_above_archived_buy_cap")
        if rule["kind"] == "healthy_pullback_range":
            if not float(rule["price_low"]) <= op <= float(rule["price_high"]):
                flags.append("open_outside_archived_pullback_range")
        elif op < float(rule["trigger_price"]):
            flags.append("open_below_archived_entry_trigger")
        episode["entry_reference_limitations"] = flags
        _event(episode, day, "open_price_reference_not_fill", price=op)
    if (float(bar["high"]) == low
            and "locked_one_price_liquidity_unproven" not in episode["execution_limitations"]):
        episode["execution_limitations"].append("locked_one_price_liquidity_unproven")
    if low <= episode["cost_stop_price"]:
        if op < episode["cost_stop_price"]:
            episode["execution_limitations"].append("gap_below_cost_stop")
        _finish(episode, day, close, "cost_8pct_stop_signal_close_reference", sessions)
        return
    if close < episode["stop_price"]:
        _finish(episode, day, close, "daily_structural_break_close_reference", sessions)
        return
    _mark(episode, day, close, sessions)
    # Same frozen trailing-structure rule as v1; never modify production stops.
    _ratchet(episode, day, history)
    if close < episode["stop_price"]:
        _finish(episode, day, close, "newly_confirmed_structural_break_close_reference", sessions)


def replay_reference(connection: sqlite3.Connection, *, sessions: list[date],
                     bars: dict[str, dict[date, dict]], through: date,
                     clearances: dict[str, dict], run_at: datetime) -> dict:
    """Record immutable statistical references using a verified session chain."""
    _aware(run_at)
    local = run_at.astimezone(SHANGHAI)
    if through > local.date() or (through == local.date() and local.time() < time(15)):
        # Even an accidentally staged daily row must not leak its eventual close
        # into an intraday run. The caller additionally verifies completed data.
        raise ValueError("reference cutoff must be a completed session before the run clock")
    if sessions != sorted(set(sessions)):
        raise ValueError("sessions must be a unique ascending verified exchange calendar")
    sessions = [day for day in sessions if day <= through]
    ensure_reference_schema(connection)
    reports = [report for report in _reports(connection)
               if date.fromisoformat(report["plan_date"]) <= through
               and datetime.fromisoformat(report["published_at"]).astimezone(UTC)
               <= run_at.astimezone(UTC)]
    isolate_methods = any(report["method_version"] == "continuous-signal-v5" for report in reports)
    replay_method = VERSION_ISOLATED_REFERENCE_METHOD if isolate_methods else REFERENCE_METHOD
    by_day: dict[date, list[tuple[dict, dict]]] = {}
    for report in reports:
        for entry in report["entries"]:
            by_day.setdefault(date.fromisoformat(report["plan_date"]), []).append((report, entry))
    episodes, active = [], {}
    for day in sorted(set(sessions) | set(by_day)):
        # A repeated preopen recommendation on the eventual exit day belongs to
        # the existing episode: its close-trigger is not knowable at publication.
        for report, entry in by_day.get(day, []):
            key = (report["method_version"] if isolate_methods else "legacy", entry["symbol"])
            old = active.get(key)
            blocked_before_entry = old and old["entry_price"] is None and old["status"] in _BLOCKED
            if old and old["status"] not in _TERMINAL and not blocked_before_entry:
                if report["report_id"] not in old["report_ids"]:
                    old["report_ids"].append(report["report_id"])
                    _event(old, day, "recommended_again_not_new_reference", report_id=report["report_id"])
                continue
            new = _episode(report, entry)
            episodes.append(new)
            active[key] = new
            if day not in sessions:
                new.update(status="evidence_missing", reason="plan_session_calendar_unverified")
        if day in sessions:
            for episode in active.values():
                _step(episode, day, bars.get(episode["symbol"], {}),
                      clearances.get(episode["symbol"], {}), sessions)
    counts = dict(Counter(episode["status"] for episode in episodes))
    closed = [episode for episode in episodes if episode["status"] == "closed"]
    active_symbols = {episode["symbol"] for episode in active.values()}
    relevant = {symbol: {day.isoformat(): bar for day, bar in sorted(history.items())
                        if day <= through}
                for symbol, history in sorted(bars.items()) if symbol in active_symbols}
    input_hash = _hash(_json({"reports": reports, "sessions": [d.isoformat() for d in sessions],
                             "bars": relevant, "clearances": clearances}))
    result = {
        "method_version": replay_method, "through": through.isoformat(),
        "reports": len(reports), "mentions": sum(len(r["entries"]) for r in reports),
        "zero_entry_reports": sum(not r["entries"] for r in reports), "episodes": episodes,
        "diagnostics": {
            "status_counts": counts, "completed_references": len(closed),
            "winning_references": sum(e["return_fraction"] > 0 for e in closed),
            "losing_references": sum(e["return_fraction"] < 0 for e in closed),
            "review_status": "insufficient_completed_samples" if len(closed) < 30 else "review_required",
            "research_flags": (["close_mark_loss_exceeds_8pct_not_a_fill"]
                               if any(e["return_fraction"] < -0.080001 for e in closed) else []),
            "rule_changes_require_user_approval": True, "rules_changed_automatically": False,
        },
        "execution_model": "recommendation_day_open_to_first_exit_trigger_day_close_reference_only",
        "execution_eligible": False, "orders_enabled": False, "holdings_modified": False,
        "conditional_entries_are_not_assumed_triggered": True,
        "comparison_with_v1_required": True, "input_hash": input_hash,
    }
    if isolate_methods:
        result["diagnostics"]["by_recommendation_method"] = {
            method: {
                "status_counts": dict(Counter(e["status"] for e in episodes
                                              if e["recommendation_method"] == method)),
                "completed_references": sum(e["recommendation_method"] == method for e in closed),
            } for method in sorted({e["recommendation_method"] for e in episodes})
        }
        result["diagnostics"]["review_status"] = "separate_method_cohorts_not_a_pooled_comparison"
    encoded = _json(result)
    result_hash = _hash(encoded)
    run_hash = _hash(replay_method + through.isoformat() + input_hash)
    with _savepoint(connection):
        old = connection.execute("SELECT result_hash FROM tracking_reference_runs WHERE run_hash=?",
                                 (run_hash,)).fetchone()
        if old and old[0] != result_hash:
            raise ValueError("same inputs produced different reference results; bump method version")
        if not old:
            connection.execute("INSERT INTO tracking_reference_runs VALUES (?,?,?,?,?,?)",
                               (run_hash, through.isoformat(), run_at.isoformat(), input_hash,
                                encoded, result_hash))
    return result


def reference_summary(connection: sqlite3.Connection) -> dict:
    ensure_reference_schema(connection)
    row = connection.execute("""SELECT result_json,result_hash FROM tracking_reference_runs
        ORDER BY through_date DESC,created_at DESC,rowid DESC LIMIT 1""").fetchone()
    if row:
        if _hash(row[0]) != row[1]:
            raise ValueError("recommendation reference integrity failure")
        return json.loads(row[0])
    reports = _reports(connection)
    return {"status": "awaiting_first_reference", "method_version": REFERENCE_METHOD,
            "reports": len(reports), "episodes": [], "execution_eligible": False,
            "diagnostics": {"rules_changed_automatically": False}}
