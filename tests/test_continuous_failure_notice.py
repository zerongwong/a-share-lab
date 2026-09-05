"""Offline final-retry alerts: no real ledger, credentials or provider calls."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ashare_lab.cli import evening_digest as cli
from ashare_lab.ports.notifications import prepare_notification_for_channel
from ashare_lab.services.build_evening_digest import EveningResearchDigest
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewRow,
    HoldingTreeReviewSummary,
)

CUTOFF = date(2026, 8, 27)
TODAY = date(2026, 8, 30)
UNSAFE_DETAIL = "PRIVATE_PROVIDER_TOKEN_MUST_NOT_ESCAPE"


class ReadOnlyRepository:
    """Only the read operation needed by the exact-revision disclosure guard."""

    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        self.reads = 0
        self.unavailable = False

    def get_current_holding_snapshot(self, *, as_of=None):
        self.reads += 1
        if self.unavailable:
            raise OSError(UNSAFE_DETAIL)
        return deepcopy(self.snapshot)


def _snapshot(*, channels=("serverchan",)):
    return {
        "revision": {
            "id": "synthetic-holdings-v1",
            "version": 1,
            "holding_weeks": 4,
            "effective_at": "2026-08-27T09:00:00+08:00",
            "source": "synthetic-test",
            "status": "active",
            "method_version": "test-only",
            "metadata_json": {"holding_summary_delivery_channels": list(channels)},
        },
        "positions": [
            {
                "id": "synthetic-position",
                "revision_id": "synthetic-holdings-v1",
                "position_key": "synthetic-key",
                "symbol": "600919",
                "name": "合成风险股",
                "entry_date": CUTOFF.isoformat(),
                "cost_price": 987654.32,
                "stock_sleeve_weight": 1.0,
                "account_weight": 0.731,
                "status": "active",
                "source": "synthetic-test",
                "version": 1,
                "metadata_json": {},
            }
        ],
    }


def _review(*, action=HoldingAction.EXIT):
    row = HoldingTreeReviewRow(
        symbol="600919",
        name="合成风险股",
        holding_weeks=4,
        holding_version=1,
        position_key="synthetic-key",
        status=HoldingReviewRowStatus.READY,
        action=action,
        latest_close=9.80,
        cost_price=987654.32,
        stock_sleeve_weight=1.0,
        account_weight=0.731,
        candidate_stop=10.10,
        previous_stop=10.00,
        effective_stop=10.10,
        stop_raised=True,
        close_below_stop=True,
        source_timeframe="daily",
        evidence_date=CUTOFF,
        slow_direction="down",
        primary_structure="breakdown",
        daily_execution="confirmed",
        reasons=("complete_close_confirmed_below_effective_stop",),
        company_action_clear=True,
        company_action_clear_from=CUTOFF,
        company_action_clear_through=CUTOFF,
    )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.READY,
        portfolio_id="synthetic-holdings-v1",
        holding_version=1,
        holding_weeks=4,
        reviewed_at=datetime(2026, 8, 30, 13, 45, tzinfo=UTC),
        data_cutoff=CUTOFF,
        rows=(row,),
    )


def _broken_builder(*_args, **_kwargs):
    raise ValueError(UNSAFE_DETAIL)


def _accepted(*channels):
    return cli.EveningNotificationSummary(
        configured_channels=channels or ("serverchan",),
        accepted_channels=channels or ("serverchan",),
        provider_receipt_ids=("synthetic-receipt",),
    )


def _paths(tmp_path: Path, *, minute=45):
    return {
        "csmar_root": tmp_path / "csmar",
        "overlay_root": tmp_path / "overlay",
        "reference_root": tmp_path / "reference",
        "state_root": tmp_path / "state",
        "log_root": tmp_path / "logs",
        "_clock": lambda: datetime(2026, 8, 30, 13, minute, tzinfo=UTC),
    }


def _run(
    tmp_path,
    *,
    repository=None,
    notifier=None,
    reviewer=_broken_builder,
    minute=45,
    builder=_broken_builder,
    latest=CUTOFF,
):
    return cli.run_evening_digest(
        **_paths(tmp_path, minute=minute),
        _repository=repository or ReadOnlyRepository(),
        _build_digest=builder,
        _build_holding_review=reviewer,
        _latest_cutoff=lambda _root: latest,
        _next_trading_day=lambda _cutoff: date(2026, 8, 28),
        _notifier=notifier or (lambda _message: _accepted()),
    )


@pytest.fixture(autouse=True)
def no_real_io(monkeypatch):
    """A missing injection is a test failure, never a network/Keychain call."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A real provider or private repository must not be used")

    monkeypatch.setattr(cli, "send_evening_digest", forbidden)
    monkeypatch.setattr(cli, "resolve_next_zero_budget_trading_day", forbidden)
    monkeypatch.setattr(cli, "SQLiteRepository", forbidden)
    monkeypatch.setattr(cli, "latest_verified_overlay_cutoff", lambda _root: CUTOFF)


@pytest.mark.parametrize("minute", [0, 15, 30, 44])
def test_earlier_error_slots_do_not_send_failure_notice(tmp_path, minute):
    messages = []
    result = _run(tmp_path, minute=minute, notifier=lambda msg: messages.append(msg))
    assert result.exit_code == cli.EXIT_ERROR
    assert not messages
    assert not (tmp_path / "state" / "evening-failure-notice-state.json").exists()


def test_final_build_error_sends_generic_once_and_records_only_acceptance(tmp_path):
    messages = []

    def notifier(message):
        messages.append(message)
        return _accepted()

    first = _run(tmp_path, notifier=notifier)
    second = _run(tmp_path, notifier=notifier)
    assert first.exit_code == second.exit_code == cli.EXIT_ERROR
    assert len(messages) == 1
    assert "未能生成或提交" in messages[0].body
    assert "请勿把旧报告当作明日买入依据" in messages[0].body
    assert "已送达" not in messages[0].body
    state = json.loads((tmp_path / "state" / "evening-failure-notice-state.json").read_text())
    assert state == {"accepted_date": TODAY.isoformat(), "delivery_confirmed": False}
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
    assert UNSAFE_DETAIL not in json.dumps(first.event)
    assert UNSAFE_DETAIL not in (tmp_path / "logs" / "evening-report.jsonl").read_text()


def test_final_retry_lock_error_also_sends_one_generic_notice(tmp_path):
    messages = []
    with daily_update_lock(tmp_path / "state" / "daily-sync.lock") as acquired:
        assert acquired
        result = _run(tmp_path, notifier=lambda msg: (messages.append(msg), _accepted())[1])
    assert result.exit_code == cli.EXIT_RETRY
    assert result.event["reason"] == "daily_data_lock_busy"
    assert len(messages) == 1


def test_full_market_failure_preserves_authorized_independent_risk_review(tmp_path):
    repo = ReadOnlyRepository(_snapshot())
    messages, calls = [], []

    def reviewer(repository, **kwargs):
        assert repository is repo
        calls.append(kwargs)
        return _review()

    def notifier(message):
        public_other_channel = prepare_notification_for_channel(message, channel_name="bark")
        assert "合成风险股" not in public_other_channel.body
        messages.append(prepare_notification_for_channel(message, channel_name="serverchan"))
        return _accepted()

    result = _run(tmp_path, repository=repo, notifier=notifier, reviewer=reviewer)
    assert result.exit_code == cli.EXIT_ERROR
    assert len(calls) == len(messages) == 1
    assert calls[0]["persist"] is False
    assert calls[0]["decision_date"] == CUTOFF
    assert calls[0]["holding_context"].version == 1
    assert "持仓单独核验" in messages[0].body
    assert "非明日新买计划" in messages[0].body
    assert "合成风险股(600919)" in messages[0].body
    assert "🔴 卖出建议" in messages[0].body
    assert "保护线10.10" in messages[0].body
    assert "987654" not in messages[0].body
    assert "0.731" not in messages[0].body
    assert "73.1" not in messages[0].body
    assert messages[0].image_url is None


@pytest.mark.parametrize("change", ["version", "authorization", "read_error"])
def test_disclosure_guard_rechecks_current_revision_and_consent(tmp_path, change):
    repo = ReadOnlyRepository(_snapshot())
    messages = []

    def notifier(message):
        assert "合成风险股" in message.body
        if change == "version":
            repo.snapshot["revision"]["version"] = 2
        elif change == "authorization":
            repo.snapshot["revision"]["metadata_json"]["holding_summary_delivery_channels"] = []
        else:
            repo.unavailable = True
        safe = prepare_notification_for_channel(message, channel_name="serverchan")
        messages.append(safe)
        return _accepted()

    _run(tmp_path, repository=repo, notifier=notifier, reviewer=lambda *_args, **_kw: _review())
    assert len(messages) == 1
    assert "合成风险股" not in messages[0].body
    assert "600919" not in messages[0].body
    assert "卖出建议" not in messages[0].body
    assert "请勿把旧报告" in messages[0].body


@pytest.mark.parametrize("channels", [(), ("bark",)])
def test_no_serverchan_holding_grant_means_generic_only(tmp_path, channels):
    messages = []

    def reviewer(*_args, **_kwargs):
        pytest.fail("ungranted holding details must not be reviewed for disclosure")

    _run(
        tmp_path,
        repository=ReadOnlyRepository(_snapshot(channels=channels)),
        reviewer=reviewer,
        notifier=lambda msg: (messages.append(msg), _accepted())[1],
    )
    assert len(messages) == 1
    assert "合成风险股" not in messages[0].body


def test_company_action_uncertainty_is_not_promoted_to_confirmed_sale(tmp_path):
    review = _review(action=HoldingAction.REVIEW)
    review = replace(
        review,
        rows=(
            replace(
                review.rows[0],
                status=HoldingReviewRowStatus.DATA_NOT_READY,
                company_action_clear=None,
                reasons=("company_action_evidence_blocks_exit:missing_independent_clearance",),
            ),
        ),
    )
    messages = []
    _run(
        tmp_path,
        repository=ReadOnlyRepository(_snapshot()),
        reviewer=lambda *_args, **_kwargs: review,
        notifier=lambda msg: (messages.append(msg), _accepted())[1],
    )
    assert len(messages) == 1
    assert "优先核验·非卖出确认" in messages[0].body
    assert "疑似破位·除权/分红待核验" in messages[0].body
    assert "参考线10.10（待核验）" in messages[0].body
    assert "🔴 卖出建议" not in messages[0].body


def test_wrong_review_revision_does_not_disclose_holdings(tmp_path):
    messages = []
    _run(
        tmp_path,
        repository=ReadOnlyRepository(_snapshot()),
        reviewer=lambda *_args, **_kwargs: replace(_review(), holding_version=2),
        notifier=lambda msg: (messages.append(msg), _accepted())[1],
    )
    assert len(messages) == 1
    assert "合成风险股" not in messages[0].body


@pytest.mark.parametrize("source", ["holding_reader", "cutoff_reader", "reviewer"])
def test_independent_review_failure_keeps_public_notice_and_sanitizes_errors(
    tmp_path,
    monkeypatch,
    source,
):
    repo = ReadOnlyRepository(_snapshot())
    messages = []
    if source == "holding_reader":
        repo.unavailable = True
    elif source == "cutoff_reader":
        monkeypatch.setattr(cli, "latest_verified_overlay_cutoff", _broken_builder)
    _run(tmp_path, repository=repo, notifier=lambda msg: (messages.append(msg), _accepted())[1])
    assert len(messages) == 1
    assert "请勿把旧报告" in messages[0].body
    assert UNSAFE_DETAIL not in messages[0].body
    assert "合成风险股" not in messages[0].body


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        True,
        "accepted",
        _accepted("bark"),
        cli.EveningNotificationSummary(
            configured_channels=("serverchan",), failed_channels=("serverchan",)
        ),
    ],
)
def test_only_explicit_serverchan_acceptance_deduplicates(tmp_path, receipt):
    messages = []
    for _ in range(2):
        _run(tmp_path, notifier=lambda msg: (messages.append(msg), receipt)[1])
    assert len(messages) == 2
    assert not (tmp_path / "state" / "evening-failure-notice-state.json").exists()


def test_provider_failure_does_not_recurse_or_replace_original_error(tmp_path):
    calls = []

    def notifier(message):
        calls.append(message)
        raise RuntimeError(UNSAFE_DETAIL)

    result = _run(tmp_path, notifier=notifier)
    assert result.exit_code == cli.EXIT_ERROR
    assert result.event["reason"] == "digest_or_provider_submission_failed"
    assert len(calls) == 1
    assert UNSAFE_DETAIL not in json.dumps(result.event)
    assert not (tmp_path / "state" / "evening-failure-notice-state.json").exists()


@pytest.mark.parametrize("with_old_state", [False, True])
def test_stale_plan_is_error_even_when_old_cutoff_was_already_accepted(tmp_path, with_old_state):
    if with_old_state:
        state_path = tmp_path / "state" / "evening-digest-state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(
            json.dumps(
                {
                    "last_provider_accepted_common_cutoff": CUTOFF.isoformat(),
                    "plan_for_date": "2026-08-28",
                    "method_version": "continuous-signal-v1",
                }
            )
        )
    digest = EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        cycle_label="合成研究",
        entry_strictness="defensive",
        max_stock_exposure=0.3,
        minimum_cash_weight=0.7,
        cycle_rule_agreement=0.8,
        periods=(),
    )
    messages, builds = [], []

    def builder(**_kwargs):
        builds.append(True)
        return digest

    result = _run(
        tmp_path, builder=builder, notifier=lambda msg: (messages.append(msg), _accepted())[1]
    )
    assert builds == [True]
    assert result.exit_code == cli.EXIT_ERROR
    assert result.event["reason"] == "verified_market_data_stale_for_tomorrow"
    assert len(messages) == 1
    assert "请勿把旧报告" in messages[0].body


def _holding_based_digest(identity):
    return EveningResearchDigest(
        common_cutoff=date(2026, 8, 28),
        decision_date=TODAY,
        cycle_label="合成研究",
        entry_strictness="defensive",
        max_stock_exposure=0.3,
        minimum_cash_weight=0.7,
        cycle_rule_agreement=0.8,
        periods=(),
        method_version="continuous-signal-v1",
        continuous_plan={
            "holding_based": True,
            "holding_identity": identity,
            "entries": [
                {
                    "symbol": "600000",
                    "name": "合成补位股",
                    "account_weight": 0.1,
                    "entry_qualified": True,
                    "entry_label": "确认≥10，买≤10.50+量",
                    "protection_line": 9.80,
                }
            ],
            "cash_weight": 0.269,
            "status_note": "私人旧仓补位计划",
            "pending_exit_symbols": [],
        },
    )


@pytest.mark.parametrize(
    "stale_identity",
    [
        ["different-holding-revision", 1],
        ["synthetic-holdings-v1", 2],
    ],
)
def test_holding_based_plan_revision_mismatch_stops_before_archive_and_submit(
    tmp_path,
    stale_identity,
):
    repo = ReadOnlyRepository(_snapshot())
    digest = _holding_based_digest(stale_identity)

    def forbidden(*_args, **_kwargs):
        pytest.fail("a plan built for another holding revision cannot be archived or submitted")

    result = cli.run_evening_digest(
        **_paths(tmp_path, minute=0),
        _repository=repo,
        _build_digest=lambda **_kwargs: digest,
        _build_holding_review=lambda *_args, **_kwargs: _review(),
        _latest_cutoff=lambda _root: digest.common_cutoff,
        _next_trading_day=lambda _cutoff: date(2026, 8, 31),
        _archive_digest=forbidden,
        _notifier=forbidden,
    )
    assert result.exit_code == cli.EXIT_ERROR
    assert result.event["reason"] == "holding_version_changed_before_submission"
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()


def test_review_failure_suppresses_replacements_but_keeps_live_authorization_guard(
    tmp_path,
    monkeypatch,
):
    repo = ReadOnlyRepository(_snapshot())
    digest = _holding_based_digest(["synthetic-holdings-v1", 1])
    messages, archives, delivery_events, reviews = [], [], [], []

    def archiver(document, repository):
        assert repository is repo
        archives.append(document)
        return SimpleNamespace(report_id="synthetic-local-archive")

    def reviewer(*_args, **_kwargs):
        reviews.append(True)
        raise ValueError(UNSAFE_DETAIL)

    def notifier(message):
        assert message.holding_authorization_guard is not None
        assert message.holding_authorization_guard("serverchan") is True
        assert "暂停发布补位" in message.body
        assert "合成补位股" not in message.body
        assert "26.9%" not in message.body
        assert "私人旧仓补位计划" not in message.body
        assert UNSAFE_DETAIL not in message.body
        repo.snapshot["revision"]["metadata_json"]["holding_summary_delivery_channels"] = []
        assert message.holding_authorization_guard("serverchan") is False
        safe = prepare_notification_for_channel(message, channel_name="serverchan")
        assert safe is not None
        assert "持仓授权待核验" in safe.body
        assert "合成补位股" not in safe.body
        assert "26.9%" not in safe.body
        assert "私人旧仓补位计划" not in safe.body
        messages.append(safe)
        return _accepted()

    monkeypatch.setattr(
        cli,
        "_record_evening_delivery_events",
        lambda *_args, **kwargs: delivery_events.append(kwargs),
    )
    result = cli.run_evening_digest(
        **_paths(tmp_path, minute=0),
        _repository=repo,
        _build_digest=lambda **_kwargs: digest,
        _build_holding_review=reviewer,
        _latest_cutoff=lambda _root: digest.common_cutoff,
        _next_trading_day=lambda _cutoff: date(2026, 8, 31),
        _archive_digest=archiver,
        _notifier=notifier,
    )
    assert result.exit_code == cli.EXIT_OK
    assert result.event["status"] == "provider_accepted"
    assert len(messages) == len(archives) == len(reviews) == len(delivery_events) == 1
