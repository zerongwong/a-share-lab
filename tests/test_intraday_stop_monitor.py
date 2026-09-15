from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ashare_lab.adapters.free_intraday_quotes import (
    IntradayQuote,
    crosscheck,
    fresh_quotes,
    parse_quotes,
    provider_code,
)
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.cli.intraday_risk import launchagent_document
from ashare_lab.services.holding_ledger import (
    HoldingPositionInput,
    clear_active_holdings,
    get_active_holding_portfolio,
    replace_active_holdings,
)
from ashare_lab.services.intraday_alert_store import confirmed_cost_touch
from ashare_lab.services.intraday_stop_monitor import run_monitor, write_private_json

CN = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 9, 14, 10, 5, tzinfo=CN)
ENTRY = date(2026, 9, 11)


@pytest.fixture
def env(tmp_path):
    repo = SQLiteRepository(
        tmp_path / "research.db", Path(__file__).resolve().parents[1] / "migrations"
    )
    repo.initialize()
    root = tmp_path / "monitor"
    write_private_json(
        root / "config.json", {"enabled": True, "authorized_channels": ["serverchan"]}
    )
    register(repo)
    return repo, root


def register(repo, *, cost=10.0, clear=True, entry=ENTRY, effective=None, name="测试股票"):
    return replace_active_holdings(
        repo,
        [
            HoldingPositionInput(
                symbol="600919",
                name=name,
                entry_date=entry,
                cost_price=cost,
                stock_sleeve_weight=1.0,
                metadata={
                    "company_action_clear": clear,
                    "company_action_clear_from": entry.isoformat(),
                    "company_action_clear_through": "2026-09-14",
                    "company_action_evidence_source": "synthetic-only",
                    "company_action_evidence_id": "test-evidence",
                },
            )
        ],
        holding_weeks=4,
        effective_at=effective or datetime(2026, 9, 11, 16, tzinfo=CN),
    )


def quotes(price=9.2, low=None, now=NOW):
    return {
        source: {
            "600919": IntradayQuote(
                "600919", source, price, low or price, max(price, 10.0), 10.0, now
            )
        }
        for source in ("tencent", "sina")
    }


def run(
    env,
    *,
    batches=None,
    now=NOW,
    notifier=None,
    calendar=lambda _d: True,
    company_action_clear_by_symbol=None,
):
    sent = []
    result = run_monitor(
        env[0],
        root=env[1],
        now=now,
        quote_fetcher=lambda _s: quotes(now=now) if batches is None else batches,
        calendar=calendar,
        notifier=notifier or (lambda msg: sent.append(msg) or True),
        company_action_clear_by_symbol=company_action_clear_by_symbol,
    )
    return result, sent


def alerts(repo, kind):
    with repo.connection() as connection:
        return connection.execute(
            "SELECT * FROM intraday_stop_alerts WHERE kind=?", (kind,)
        ).fetchall()


def test_exact_eight_percent_is_confirmed_and_sent_once_without_orders(env):
    before = get_active_holding_portfolio(env[0])
    event, sent = run(env)
    assert any("已触及8%" in m.body for m in sent)
    assert "10.00" not in "\n".join(m.body for m in sent)
    assert len(alerts(env[0], "cost_exit")) == 1
    event2, sent2 = run(env, now=NOW + timedelta(minutes=1))
    assert event2["accepted"] == 0 and not sent2
    assert event["orders_enabled"] is False
    assert get_active_holding_portfolio(env[0]) == before
    assert (
        confirmed_cost_touch(
            env[0], before.positions[0].position_key, cutoff=NOW.date(), known_at=NOW
        )
        == NOW.date()
    )


def test_intraday_day_low_catches_between_poll_touch_for_old_holding(env):
    _, sent = run(env, batches=quotes(9.5, low=9.19))
    assert any("已触及8%" in m.body for m in sent)


def test_matching_last_prices_do_not_confirm_a_single_source_low(env):
    data = quotes(9.5, low=9.1)
    data["sina"] = quotes(9.5, low=9.4)["sina"]
    _, sent = run(env, batches=data)
    assert not alerts(env[0], "cost_exit")
    assert alerts(env[0], "cost_review")
    assert any("疑似触及8%" in m.body for m in sent)


def test_future_effective_holding_is_not_monitored_before_confirmation_time(env):
    register(env[0], effective=NOW + timedelta(minutes=1))
    event = run_monitor(
        env[0],
        root=env[1],
        now=NOW,
        quote_fetcher=lambda _s: pytest.fail("holding not effective yet"),
        calendar=lambda _d: pytest.fail("holding not effective yet"),
        notifier=lambda _m: pytest.fail("holding not effective yet"),
    )
    assert event["status"] == "holding_not_yet_effective"


def test_today_entry_low_may_precede_purchase_and_does_not_trigger(env):
    register(env[0], entry=NOW.date(), effective=NOW - timedelta(minutes=5))
    _, sent = run(env, batches=quotes(9.5, low=9.1))
    assert not alerts(env[0], "cost_exit")
    assert all("已触及8%" not in m.body for m in sent)


@pytest.mark.parametrize("clear,source_error", [(False, False), (True, True)])
def test_missing_evidence_or_single_source_prompts_review_not_confirmed_exit(
    env, clear, source_error
):
    register(env[0], clear=clear)
    data = quotes(9.1)
    if source_error:
        data["sina"] = {}
    _, sent = run(env, batches=data)
    assert alerts(env[0], "cost_review") and not alerts(env[0], "cost_exit")
    assert any("疑似触及8%" in m.body for m in sent)


def test_archived_independent_clearance_confirms_without_manual_metadata(env):
    from ashare_lab.services.review_active_holdings import CompanyActionClearance

    register(env[0], clear=False)
    clearance = CompanyActionClearance(
        symbol="600919",
        through_date=NOW.date(),
        clear=True,
        source="cninfo",
        evidence_id="cninfo:test",
        from_date=ENTRY,
    )

    _, sent = run(
        env,
        batches=quotes(9.1),
        company_action_clear_by_symbol={"600919": clearance},
    )

    assert alerts(env[0], "cost_exit")
    assert any("已触及8%" in message.body for message in sent)


def test_detected_independent_company_action_blocks_manual_clear_flag(env):
    from ashare_lab.services.review_active_holdings import CompanyActionClearance

    detected = CompanyActionClearance(
        symbol="600919",
        through_date=NOW.date(),
        clear=False,
        source="cninfo",
        evidence_id="cninfo:detected",
        from_date=ENTRY,
    )
    _, sent = run(
        env,
        batches=quotes(9.1),
        company_action_clear_by_symbol={"600919": detected},
    )
    assert alerts(env[0], "cost_review") and not alerts(env[0], "cost_exit")
    assert any("公司行动" in message.body or "除权" in message.body for message in sent)


def test_stale_quotes_cannot_generate_a_stop_confirmation(env):
    event, sent = run(env, batches=quotes(8.0, now=NOW - timedelta(minutes=4)))
    assert event["status"] == "degraded"
    assert not alerts(env[0], "cost_exit")
    assert any("过期" in m.body for m in sent)


def test_conflicting_sources_do_not_create_confirmed_exit(env):
    data = quotes(9.1)
    data["sina"] = quotes(10.0)["sina"]
    run(env, batches=data)
    assert not alerts(env[0], "cost_exit")
    assert alerts(env[0], "cost_review")


@pytest.mark.parametrize(
    "now",
    [
        NOW.replace(hour=8),
        NOW.replace(hour=12),
        NOW.replace(hour=16),
        NOW.replace(day=12),
        NOW.replace(day=13),
    ],
)
def test_outside_trading_hours_is_quiet_without_network(env, now):
    event = run_monitor(
        env[0],
        root=env[1],
        now=now,
        quote_fetcher=lambda _s: pytest.fail("no quote request"),
        calendar=lambda _d: pytest.fail("no calendar request"),
        notifier=lambda _m: pytest.fail("no notification"),
    )
    assert event["status"] == "outside_session"


def test_empty_portfolio_has_no_quotes_or_alerts(env):
    clear_active_holdings(env[0], effective_at=NOW)
    event = run_monitor(
        env[0],
        root=env[1],
        now=NOW,
        quote_fetcher=lambda _s: pytest.fail("empty must not fetch"),
        calendar=lambda _d: pytest.fail("empty must not fetch"),
        notifier=lambda _m: pytest.fail("empty must be quiet"),
    )
    assert event["status"] == "no_holdings"


def test_verified_holiday_is_quiet(env):
    event, sent = run(env, calendar=lambda _d: False)
    assert event["status"] == "market_closed" and not sent


def test_calendar_failure_is_alerted_without_guessing_session(env):
    event, sent = run(env, calendar=lambda _d: None)
    assert event["status"] == "calendar_unavailable"
    assert any("交易日历" in m.body for m in sent)


def test_missing_cost_is_explicitly_reported(env):
    register(env[0], cost=None)
    _, sent = run(env)
    assert any("未登记有效成本" in m.body for m in sent)
    assert not alerts(env[0], "cost_exit")


def test_failed_delivery_retries_after_three_minutes_and_is_bounded(env):
    def failed(_m):
        return False

    event, _ = run(env, notifier=failed)
    assert event["failed"] > 0
    event, _ = run(env, now=NOW + timedelta(minutes=1), notifier=failed)
    assert event["failed"] == 0
    for minutes in (3, 6, 9, 12):
        run(env, now=NOW + timedelta(minutes=minutes), notifier=failed)
    assert alerts(env[0], "cost_exit")[0]["attempts"] == 3


def test_holding_change_during_quote_fetch_cancels_old_disclosure(env):
    def fetch(_symbols):
        clear_active_holdings(env[0], effective_at=NOW)
        return quotes()

    event = run_monitor(
        env[0],
        root=env[1],
        now=NOW,
        quote_fetcher=fetch,
        calendar=lambda _d: True,
        notifier=lambda _m: pytest.fail("stale holding leaked"),
    )
    assert event["status"] == "holding_or_authorization_changed"
    assert not alerts(env[0], "cost_exit")


def test_last_moment_guard_rejects_after_clear(env):
    def notifier(message):
        clear_active_holdings(env[0], effective_at=NOW)
        assert not message.holding_authorization_guard("serverchan")
        return False

    run(env, notifier=notifier)


def test_disabled_monitor_sends_nothing(env):
    write_private_json(
        env[1] / "config.json", {"enabled": False, "authorized_channels": ["serverchan"]}
    )
    event, sent = run(env)
    assert event["status"] == "disabled" and not sent


def test_future_intraday_observation_does_not_leak_into_prior_close(env):
    run(env)
    holding = get_active_holding_portfolio(env[0]).positions[0]
    assert confirmed_cost_touch(env[0], holding.position_key, cutoff=ENTRY, known_at=NOW) is None
    assert (
        confirmed_cost_touch(
            env[0], holding.position_key, cutoff=NOW.date(), known_at=NOW - timedelta(seconds=1)
        )
        is None
    )


def test_parser_uses_provider_timestamp_not_retrieval_time():
    fields = [""] * 35
    fields[1:6] = ["示例", "600919", "9.2", "10", "9.3"]
    fields[30], fields[33], fields[34] = "20260914100500", "9.5", "9.1"
    parsed = parse_quotes('v_sh600919="' + "~".join(fields) + '";', "tencent", ("600919",))
    assert parsed["600919"].quoted_at == NOW
    assert parsed["600919"].price == 9.2
    assert not fresh_quotes(tuple(parsed.values()), now=NOW + timedelta(minutes=2))


def test_sina_parser_and_symbol_scope():
    fields = [""] * 32
    fields[:6] = ["示例", "9.3", "10", "9.2", "9.5", "9.1"]
    fields[30:] = ["2026-09-14", "10:05:00"]
    parsed = parse_quotes('var hq_str_sh600919="' + ",".join(fields) + '";', "sina", ("600919",))
    assert parsed["600919"].quoted_at == NOW
    assert provider_code("000001") == "sz000001"  # Stock, not Shanghai index.
    assert provider_code("920001") == "bj920001"
    with pytest.raises(ValueError):
        provider_code("100001")


def test_pair_check_rejects_future_stale_and_different_prices():
    data = quotes()
    pair = tuple(b["600919"] for b in data.values())
    assert crosscheck(pair)
    assert not fresh_quotes(pair, now=NOW - timedelta(minutes=2))
    assert not crosscheck(pair[:1])


def test_worker_is_independent_and_runs_every_minute():
    doc = launchagent_document("/path/.venv/bin/python", "/project")
    assert doc["StartInterval"] == 60
    assert doc["ProgramArguments"][-2:] == ["ashare_lab.cli.intraday_risk", "worker"]
    assert "daily-sync" not in str(doc)


def test_structure_touch_is_immediate_but_not_a_confirmed_structural_exit(env):
    repo, _ = env
    holding = get_active_holding_portfolio(repo).positions[0]
    with repo.connection() as connection:
        connection.execute(
            """INSERT INTO holding_protective_stops
               (position_key, symbol, entry_date, effective_stop, candidate_stop, data_cutoff,
                source_timeframe, evidence_date, holding_version, method_version, details_json, updated_at)
               VALUES (?, '600919', '2026-09-11', 9.8, 9.8, '2026-09-11', 'daily',
                       '2026-09-11', 1, 'synthetic', '{}', ?)""",
            (holding.position_key, NOW.isoformat()),
        )
    _, sent = run(env, batches=quotes(9.7))
    assert alerts(repo, "structure_touch") and not alerts(repo, "cost_exit")
    assert any("等待收盘确认" in msg.body for msg in sent)
    assert repo.get_holding_protective_stop(holding.position_key)["effective_stop"] == 9.8


def test_confirmed_entry_day_touch_survives_eod_rebound_and_missing_data(env):
    from test_review_active_holdings import _history

    from ashare_lab.services.review_active_holdings import (
        CompanyActionClearance,
        HoldingAction,
        review_active_holdings,
    )

    register(env[0], entry=NOW.date(), effective=NOW - timedelta(minutes=10))
    run(env, batches=quotes(9.1))
    history = _history(end=NOW.date().isoformat())
    history.loc[history.index[-1], ["open", "high", "low", "close"]] = [10.0, 10.1, 9.1, 9.5]
    kwargs = {
        "as_of": NOW.date(),
        "reviewed_at": NOW.replace(hour=17),
        "continuous_profile": True,
        "persist": False,
    }
    summary = review_active_holdings(
        env[0],
        {"600919": history},
        company_action_clear_by_symbol={
            "600919": CompanyActionClearance(
                symbol="600919",
                through_date=NOW.date(),
                from_date=NOW.date(),
                clear=True,
                source="synthetic-only",
                evidence_id="verified-test",
            )
        },
        **kwargs,
    )
    assert summary.rows[0].action is HoldingAction.EXIT
    assert summary.rows[0].cost_stop_touched_on == NOW.date()
    no_data = review_active_holdings(env[0], {}, **kwargs)
    assert no_data.rows[0].urgent and no_data.rows[0].action is HoldingAction.REVIEW


def test_same_check_combines_risk_and_health_in_one_serverchan_message(env):
    event, sent = run(env)
    assert len(sent) == 1
    assert event["alert_count"] == 2
    assert event["accepted"] == 1


def test_confirmed_unsent_cost_risk_retries_hourly_after_initial_attempts(env):
    for minutes in (0, 3, 6):
        run(env, now=NOW + timedelta(minutes=minutes), notifier=lambda _m: False)
    event, sent = run(
        env, now=NOW + timedelta(minutes=70), batches=quotes(9.5, now=NOW + timedelta(minutes=70))
    )
    assert sent and event["accepted"] == 1
    assert alerts(env[0], "cost_exit")[0]["attempts"] == 4


def test_activation_test_is_explicit_and_not_repeated_when_delivery_is_uncertain(env, monkeypatch):
    from ashare_lab.cli import intraday_risk

    sent = []
    monkeypatch.setattr(intraday_risk, "_root", lambda: env[1])
    monkeypatch.setattr(intraday_risk, "send_serverchan", lambda m: sent.append(m) or False)
    first = intraday_risk.test_notification()
    assert first["status"] == "delivery_unconfirmed"
    assert intraday_risk.test_notification() == first
    assert len(sent) == 1 and "启用测试" in sent[0].body


def test_disabled_monitor_cannot_send_activation_test(env, monkeypatch):
    from ashare_lab.cli import intraday_risk

    write_private_json(env[1] / "config.json", {"enabled": False})
    monkeypatch.setattr(intraday_risk, "_root", lambda: env[1])
    monkeypatch.setattr(intraday_risk, "send_serverchan", lambda _m: pytest.fail("disabled"))
    assert intraday_risk.test_notification()["status"] == "disabled_or_already_running"


def test_worker_error_records_current_failure_and_attempts_generic_notice(env, monkeypatch):
    import json
    from types import SimpleNamespace

    from ashare_lab.cli import intraday_risk

    notices = []
    monkeypatch.setattr(intraday_risk, "_root", lambda: env[1])
    monkeypatch.setattr(
        intraday_risk.subprocess, "Popen", lambda *_a, **_kw: SimpleNamespace(wait=lambda **_kw: 2)
    )
    monkeypatch.setattr(intraday_risk.subprocess, "run", lambda args, **_kw: notices.append(args))
    assert intraday_risk.supervise() == 2
    assert json.loads((env[1] / "last-status.json").read_text())["status"] == "monitor_worker_error"
    assert notices[0][-1] == "failure-notice"
