from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from ashare_lab.services.intraday_stop_monitor import write_private_json
from ashare_lab.services.recommendation_exit_notifications import CN, dispatch_reference_exits
from ashare_lab.services.recommendation_reference import REFERENCE_METHOD

NOW = datetime(2026, 10, 14, 15, 40, tzinfo=CN)


@pytest.fixture
def root(tmp_path):
    write_private_json(
        tmp_path / "config.json",
        {
            "enabled": True,
            "authorized_channels": ["serverchan"],
            "allow_reference_exit_notifications": True,
        },
    )
    return tmp_path


def summary():
    return {
        "method_version": REFERENCE_METHOD,
        "through": "2026-10-14",
        "episodes": [
            {
                "episode_id": "one",
                "symbol": "600001",
                "name": "合成样本",
                "status": "closed",
                "entry_date": "2026-10-08",
                "exit_date": "2026-10-14",
                "entry_price": 10.0,
                "exit_price": 9.0,
                "trading_days": 5,
                "profit_per_share": -1.0,
                "return_fraction": -0.1,
                "daily_average_return": -0.02,
                "daily_average_profit_per_share": -0.2,
            }
        ],
    }


def test_closed_reference_notifies_once_and_shows_per_share_not_fictional_capital(root):
    sent = []

    def notify(msg):
        sent.append(msg)
        return True

    result = dispatch_reference_exits(summary=summary(), root=root, notifier=notify, now=NOW)
    assert result["accepted"] == 1 and result["delivery_confirmed"] is False
    body = sent[0].body
    for text in (
        "5 个交易日",
        "-10.00%",
        "-2.00%",
        "-0.2000 元/股/日",
        "未计费用分红",
        "不代表买点条件已触发",
    ):
        assert text in body
    assert sent[0].holding_authorization_guard("bark") is False
    assert (
        dispatch_reference_exits(summary=summary(), root=root, notifier=notify, now=NOW)["accepted"]
        == 0
    )
    assert len(sent) == 1


def test_method_isolation_upgrade_does_not_resend_accepted_episode(root):
    from ashare_lab.services.recommendation_reference import VERSION_ISOLATED_REFERENCE_METHOD

    report = summary()
    first = dispatch_reference_exits(summary=report, root=root, notifier=lambda _: True, now=NOW)
    assert first["accepted"] == 1
    report["method_version"] = VERSION_ISOLATED_REFERENCE_METHOD
    sent = []
    second = dispatch_reference_exits(
        summary=report, root=root, notifier=lambda msg: sent.append(msg) or True, now=NOW,
    )
    assert second["status"] == "no_new_completed_references"
    assert sent == []


def test_failure_retries_bounded_and_unknown_response_not_accepted(root):
    sent = []

    def notify(msg):
        sent.append(msg)
        raise TimeoutError("private response must not be logged")

    assert (
        dispatch_reference_exits(summary=summary(), root=root, notifier=notify, now=NOW)["status"]
        == "provider_not_accepted"
    )
    dispatch_reference_exits(
        summary=summary(), root=root, notifier=notify, now=NOW + timedelta(minutes=1)
    )
    assert len(sent) == 1
    result = dispatch_reference_exits(
        summary=summary(), root=root, notifier=lambda _: True, now=NOW + timedelta(minutes=3)
    )
    assert result["accepted"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 1234.0])
def test_invalid_or_inconsistent_metrics_never_send(root, bad):
    report = summary()
    report["episodes"][0]["daily_average_return"] = bad
    result = dispatch_reference_exits(
        summary=report, root=root, notifier=lambda _: pytest.fail("must not send"), now=NOW
    )
    assert result["status"] == "reference_integrity_error"


def test_nonclosed_references_and_disabled_authorization_are_quiet(root):
    report = summary()
    report["episodes"][0]["status"] = "open"

    def no_send(_):
        pytest.fail("must not send")

    assert (
        dispatch_reference_exits(summary=report, root=root, notifier=no_send, now=NOW)["accepted"]
        == 0
    )
    write_private_json(root / "config.json", {"enabled": False})
    assert (
        dispatch_reference_exits(summary=summary(), root=root, notifier=no_send, now=NOW)["status"]
        == "disabled"
    )


def test_corrupt_receipt_is_not_reset_and_resent(root):
    (root / "delivery-state.json").write_text("broken", encoding="utf-8")
    result = dispatch_reference_exits(
        summary=summary(), root=root, notifier=lambda _: pytest.fail("must not send"), now=NOW
    )
    assert result["status"] == "delivery_state_needs_review"


def test_changed_previously_accepted_reference_requires_review(root):
    report = summary()
    dispatch_reference_exits(summary=report, root=root, notifier=lambda _: True, now=NOW)
    report["episodes"][0]["name"] = "更正样本"
    result = dispatch_reference_exits(
        summary=report, root=root, notifier=lambda _: pytest.fail("must not send"), now=NOW
    )
    assert result["status"] == "accepted_reference_changed_needs_review"


def test_receipt_persists_before_provider_and_does_not_include_secrets(root):
    def notifier(_):
        state = json.loads((root / "delivery-state.json").read_text())
        record = next(iter(state.values()))
        assert record["attempts"] == 1 and record["accepted"] is False
        assert "entry_price" not in record
        return True

    assert (
        dispatch_reference_exits(summary=summary(), root=root, notifier=notifier, now=NOW)[
            "accepted"
        ]
        == 1
    )


def test_sync_hook_uses_only_authorized_reference_and_sanitized_status(root, monkeypatch):
    from ashare_lab.cli import holding_pnl, scheduled_sync

    sent = []
    monkeypatch.setattr(holding_pnl, "send_serverchan", lambda msg: sent.append(msg) or True)
    event = {}
    scheduled_sync._notify_reference_completions(event, root=root, summary=summary(), now=NOW)
    assert event["recommendation_reference_notice_status"] == "provider_accepted"
    assert set(event) == {
        "recommendation_reference_notice_status",
        "recommendation_reference_notices_accepted",
    }
    assert len(sent) == 1


def test_same_day_close_is_never_announced_before_close(root):
    result = dispatch_reference_exits(
        summary=summary(),
        root=root,
        notifier=lambda _: pytest.fail("must not send"),
        now=NOW.replace(hour=9, minute=0),
    )
    assert result["status"] == "reference_integrity_error"


def test_duplicate_episode_is_an_integrity_failure(root):
    report = summary()
    report["episodes"].append(dict(report["episodes"][0]))
    result = dispatch_reference_exits(
        summary=report,
        root=root,
        notifier=lambda _: pytest.fail("must not send"),
        now=NOW,
    )
    assert result["status"] == "reference_integrity_error"


@pytest.mark.parametrize(
    "new_status", ["evidence_missing", "corporate_action_review_required", "absent"]
)
def test_previously_announced_reference_losing_evidence_is_not_silently_ignored(root, new_status):
    report = summary()
    dispatch_reference_exits(summary=report, root=root, notifier=lambda _: True, now=NOW)
    if new_status == "absent":
        report["episodes"] = []
    else:
        report["episodes"][0]["status"] = new_status
    result = dispatch_reference_exits(
        summary=report,
        root=root,
        notifier=lambda _: pytest.fail("must not resend"),
        now=NOW,
    )
    assert result["status"] == "accepted_reference_changed_needs_review"
