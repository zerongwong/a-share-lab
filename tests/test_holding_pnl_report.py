from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ashare_lab.adapters.free_intraday_quotes import IntradayQuote
from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.ports.notifications import prepare_notification_for_channel
from ashare_lab.services.holding_ledger import (
    HoldingPositionInput,
    clear_active_holdings,
    get_active_holding_portfolio,
    replace_active_holdings,
)
from ashare_lab.services.holding_pnl_report import (
    CN,
    build_holding_pnl_report,
    render_holding_pnl_report,
    run_holding_pnl,
    send_unverified_holding_pnl,
)
from ashare_lab.services.intraday_stop_monitor import write_private_json
from ashare_lab.services.review_active_holdings import CompanyActionClearance

NOW = datetime(2026, 9, 22, 15, 35, tzinfo=CN)
ENTRY = date(2026, 9, 21)


@pytest.fixture
def env(tmp_path):
    repo = SQLiteRepository(tmp_path / "research.db", Path(__file__).parents[1] / "migrations")
    repo.initialize()
    root = tmp_path / "pnl"
    write_private_json(
        root / "config.json",
        {
            "enabled": True,
            "authorized_channels": ["serverchan"],
            "allow_pnl_amounts": True,
        },
    )
    replace_active_holdings(
        repo,
        [
            HoldingPositionInput(
                symbol=symbol,
                name=name,
                entry_date=ENTRY,
                cost_price=cost,
                stock_sleeve_weight=0.5,
                metadata={"quantity": quantity, "quantity_user_confirmed": True},
            )
            for symbol, name, cost, quantity in [
                ("600919", "甲股票", 10.0, 100),
                ("601298", "乙股票", 20.0, 200),
            ]
        ],
        holding_weeks=4,
        effective_at=NOW - timedelta(days=1),
    )
    return repo, root


def batches(now=NOW):
    return {
        source: {
            symbol: IntradayQuote(
                symbol,
                source,
                price,
                price - 1,
                price + 1,
                prior,
                now.replace(hour=15, minute=0, second=3),
            )
            for symbol, price, prior in [("600919", 11.0, 10.5), ("601298", 19.0, 19.5)]
        }
        for source in ("tencent", "sina")
    }


def clearances(now=NOW):
    return {
        symbol: CompanyActionClearance(
            symbol=symbol,
            through_date=now.date(),
            clear=True,
            source="official-synthetic",
            evidence_id="synthetic-id",
            from_date=ENTRY,
            knowledge_time=now - timedelta(minutes=1),
        )
        for symbol in ("600919", "601298")
    }


def build(env, *, quotes=None, evidence=None, portfolio=None):
    return build_holding_pnl_report(
        portfolio or get_active_holding_portfolio(env[0]),
        now=NOW,
        batches=batches() if quotes is None else quotes,
        clearances=clearances() if evidence is None else evidence,
    )


def run(env, *, now=NOW, quotes=None, evidence=None, notifier=None, calendar=None, clock=None):
    messages, requests = [], []

    def quote_fetcher(symbols):
        requests.append(symbols)
        return batches(now) if quotes is None else quotes

    event = run_holding_pnl(
        env[0],
        root=env[1],
        now=now,
        clock=clock or (lambda: now),
        quote_fetcher=quote_fetcher,
        calendar=calendar or (lambda _: True),
        notifier=notifier or (lambda message: messages.append(message) or True),
        company_action_clearance_loader=lambda *_args, **_kwargs: (
            clearances(now) if evidence is None else evidence
        ),
    )
    return event, messages, requests


def test_actual_quantity_cost_weighted_floating_pnl_not_daily_or_simple_mean(env):
    report = build(env)
    assert report.complete
    assert report.rows[0].pnl_amount == Decimal("100.0")
    assert report.rows[0].pnl_percent == Decimal("10")
    assert report.rows[1].pnl_amount == Decimal("-200")
    assert report.pnl_amount == Decimal("-100")
    assert report.pnl_percent == Decimal("-2")
    body = render_holding_pnl_report(report)
    assert "持仓合计：-2.00%｜-100.00元" in body
    assert "非当日盈亏、非全账户" in body
    assert "未计未录入费用、分红及已清仓收益" in body
    assert "非交易所正式日线" in body
    assert "5,000" not in body and "100股" not in body and "成本10" not in body


@pytest.mark.parametrize(
    "change",
    [
        {"quoted_at": NOW - timedelta(days=1)},
        {"quoted_at": NOW.replace(hour=14, minute=59)},
        {"quoted_at": NOW + timedelta(seconds=1)},
        {"quoted_at": NOW.replace(tzinfo=None)},
        {"price": 11.02},
        {"price": 11.01},
        {"price": float("nan")},
        {"symbol": "000001"},
        {"source": "tencent"},
        {"previous_close": 10.501},
        {"low": 12.0},
    ],
)
def test_bad_close_or_identity_never_enters_pnl(env, change):
    quotes = batches()
    quotes["sina"]["600919"] = replace(quotes["sina"]["600919"], **change)
    report = build(env, quotes=quotes)
    assert not report.complete
    assert report.rows[0].pnl_amount is None and report.rows[0].pnl_percent is None
    assert report.pnl_amount is None and report.pnl_percent is None


def test_single_source_is_not_eod_confirmation(env):
    report = build(env, quotes={"tencent": batches()["tencent"]})
    assert not report.complete
    assert all(row.pnl_percent is None for row in report.rows)


@pytest.mark.parametrize(
    "changes",
    [
        {"clear": False},
        {"clear": "unknown"},
        {"clear": 1},
        {"knowledge_time": None},
        {"knowledge_time": NOW + timedelta(seconds=1)},
        {"from_date": ENTRY + timedelta(days=1)},
        {"through_date": ENTRY},
        {"through_date": NOW.date() + timedelta(days=1)},
        {"symbol": "000001"},
        {"evidence_id": ""},
    ],
)
def test_company_action_unknown_or_stale_blocks_comparable_return(env, changes):
    evidence = clearances()
    evidence["600919"] = replace(evidence["600919"], **changes)
    report = build(env, evidence=evidence)
    assert report.rows[0].pnl_percent is None
    assert "除权分红核验未完成" in report.rows[0].issues
    assert report.pnl_amount is None


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"quantity": 100},
        {"quantity": 0, "quantity_user_confirmed": True},
        {"quantity": True, "quantity_user_confirmed": True},
        {"quantity": 100.0, "quantity_user_confirmed": True},
    ],
)
def test_missing_confirmed_quantity_allows_percent_only_not_partial_portfolio(env, metadata):
    portfolio = get_active_holding_portfolio(env[0])
    portfolio = replace(
        portfolio,
        positions=(replace(portfolio.positions[0], metadata=metadata), portfolio.positions[1]),
    )
    report = build(env, portfolio=portfolio)
    assert report.rows[0].pnl_percent == Decimal(10)
    assert report.rows[0].pnl_amount is None
    assert report.pnl_amount is None and report.pnl_percent is None
    assert "股数待补齐" in render_holding_pnl_report(report)


def test_cost_missing_does_not_infer_from_prevclose_or_weight(env):
    portfolio = get_active_holding_portfolio(env[0])
    portfolio = replace(
        portfolio,
        positions=(replace(portfolio.positions[0], cost_price=None), portfolio.positions[1]),
    )
    report = build(env, portfolio=portfolio)
    assert report.rows[0].pnl_percent is None
    assert report.pnl_amount is None


def test_success_deduplicates_without_changing_ledger_or_disclosing_capital(env):
    original = get_active_holding_portfolio(env[0])
    event, messages, requests = run(env)
    assert event["status"] == "provider_accepted"
    assert event["delivery_confirmed"] is False
    assert requests == [("600919", "601298")]
    assert messages[0].holding_authorization_guard("serverchan") is True
    assert messages[0].holding_authorization_guard("bark") is False
    assert get_active_holding_portfolio(env[0]) == original
    assert run(env)[0]["status"] == "already_provider_accepted"
    state = (env[1] / "delivery-state.json").read_text()
    assert "600919" not in state and "100" not in state


def test_partial_only_sends_last_slot_as_same_daily_report(env):
    assert run(env, evidence={})[0]["status"] == "awaiting_verified_pnl"
    assert run(env, evidence={})[1] == []
    event, messages, _ = run(env, now=NOW.replace(minute=55), evidence={})
    assert event["status"] == "pending_provider_accepted"
    assert "收盘盈亏待核验" in messages[0].title
    assert "除权分红核验未完成" in messages[0].body
    assert "10.00%" not in messages[0].body
    assert run(env, now=NOW.replace(minute=56))[0]["status"] == "already_provider_accepted"


def test_unknown_calendar_final_notice_contains_no_holdings_or_pnl(env):
    event, messages, requests = run(env, now=NOW.replace(minute=55), calendar=lambda _: None)
    assert event["status"] == "pending_provider_accepted"
    assert requests == []
    assert "今日一定开市" in messages[0].body
    assert "600919" not in messages[0].body and "甲股票" not in messages[0].body


def test_closed_session_never_sends_or_fetches(env):
    event, messages, requests = run(env, now=NOW.replace(minute=55), calendar=lambda _: False)
    assert event["status"] == "market_closed"
    assert not messages and not requests


def test_empty_portfolio_never_sends_or_fetches_old_holdings(env):
    clear_active_holdings(env[0], effective_at=NOW)
    event, messages, requests = run(env)
    assert event["status"] == "no_holdings"
    assert not messages and not requests
    later = NOW.replace(minute=55)
    assert (
        send_unverified_holding_pnl(
            env[0],
            root=env[1],
            clock=lambda: later,
            notifier=lambda _: pytest.fail("empty positions never receive fallback"),
        )["status"]
        == "no_holdings"
    )


def test_missing_explicit_amount_grant_blocks_old_summary_grant(env):
    write_private_json(
        env[1] / "config.json", {"enabled": True, "authorized_channels": ["serverchan"]}
    )
    event, messages, requests = run(env)
    assert event["status"] == "disabled"
    assert not messages and not requests


def test_changed_revision_or_revoked_amount_grant_strips_sensitive_body(env):
    def notifier(message):
        clear_active_holdings(env[0], effective_at=NOW)
        safe = prepare_notification_for_channel(message, channel_name="serverchan")
        assert "600919" not in safe.body and "+100" not in safe.body
        assert message.holding_authorization_guard("serverchan") is False
        return True

    event, _, _ = run(env, notifier=notifier)
    assert event["status"] == "holding_or_authorization_changed"
    assert not (env[1] / "delivery-state.json").exists()


def test_quote_read_finishes_after_deadline_never_sends(env):
    moments = iter([NOW, NOW, NOW, NOW, NOW.replace(hour=16, minute=0)])
    last = NOW

    def clock():
        nonlocal last
        last = next(moments, last)
        return last

    event, messages, _ = run(env, clock=clock)
    assert event["status"] == "holding_or_authorization_changed"
    assert not messages


def test_final_fallback_uses_same_receipt_and_contains_no_private_fields(env):
    later = NOW.replace(minute=55)
    messages = []
    result = send_unverified_holding_pnl(
        env[0],
        root=env[1],
        clock=lambda: later,
        notifier=lambda message: messages.append(message) or True,
    )
    assert result["status"] == "pending_provider_accepted"
    assert "600919" not in messages[0].body and "%" not in messages[0].body
    assert "不是盈亏为零" in messages[0].body
    assert run(env, now=later)[0]["status"] == "already_provider_accepted"


def test_provider_rejects_does_not_mark_delivery(env):
    event, _, _ = run(env, notifier=lambda _: False)
    assert event["status"] == "provider_not_accepted"
    assert not (env[1] / "delivery-state.json").exists()


def test_final_fallback_suppresses_verified_market_holiday(env):
    write_private_json(env[1] / "calendar.json", {"date": NOW.date().isoformat(), "open": False})
    result = send_unverified_holding_pnl(
        env[0],
        root=env[1],
        clock=lambda: NOW.replace(minute=55),
        notifier=lambda _: pytest.fail("verified holiday must stay quiet"),
    )
    assert result["status"] == "market_closed"


def test_revoking_config_after_report_built_denies_disclosure(env):
    def notifier(message):
        write_private_json(
            env[1] / "config.json",
            {
                "enabled": False,
                "authorized_channels": ["serverchan"],
                "allow_pnl_amounts": True,
            },
        )
        assert message.holding_authorization_guard("serverchan") is False
        safe = prepare_notification_for_channel(message, channel_name="serverchan")
        assert "600919" not in safe.body and "+100" not in safe.body
        return True

    assert run(env, notifier=notifier)[0]["status"] == "holding_or_authorization_changed"


def test_unverified_fallback_cannot_duplicate_completed_report(env):
    run(env)
    result = send_unverified_holding_pnl(
        env[0],
        root=env[1],
        clock=lambda: NOW.replace(minute=55),
        notifier=lambda _: pytest.fail("completed daily receipt is shared with fallback"),
    )
    assert result["status"] == "already_provider_accepted"
