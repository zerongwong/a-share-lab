from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.cli import evening_report
from ashare_lab.cli.evening_digest import _message_for_channel
from ashare_lab.domain.errors import NotificationDeliveryError
from ashare_lab.ports.notifications import NotificationReceipt
from ashare_lab.services.build_evening_digest import EveningResearchDigest
from ashare_lab.services.build_holding_chart_report import (
    HoldingChartBuildResult,
    HoldingChartBuildStatus,
)
from ashare_lab.services.daily_update_lock import daily_update_lock
from ashare_lab.services.holding_ledger import (
    HOLDING_CHART_DELIVERY_CHANNELS_KEY,
    HOLDING_CHART_PUBLISHER_ID_KEY,
    ActiveHoldingPortfolio,
    HoldingPositionInput,
    replace_active_holdings,
)
from ashare_lab.services.review_active_holdings import (
    HoldingAction,
    HoldingReviewRowStatus,
    HoldingReviewSummaryStatus,
    HoldingTreeReviewRow,
    HoldingTreeReviewSummary,
)

CUTOFF = date(2026, 8, 27)
FRIDAY = date(2026, 8, 28)
SATURDAY = date(2026, 8, 29)
SUNDAY = date(2026, 8, 30)


def _digest() -> EveningResearchDigest:
    return EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        cycle_label="中期下行｜短线修复反弹",
        entry_strictness="defensive",
        max_stock_exposure=0.30,
        minimum_cash_weight=0.70,
        cycle_rule_agreement=0.875,
        periods=(),
    )


def _paths(tmp_path: Path) -> dict[str, object]:
    return {
        "csmar_root": tmp_path / "csmar",
        "overlay_root": tmp_path / "overlay",
        "reference_root": tmp_path / "reference",
        "state_root": tmp_path / "state",
        "log_root": tmp_path / "logs",
        "_clock": lambda: datetime.now(UTC).replace(hour=13, minute=0, second=0),
    }


def _accepted_summary(
    *channels: str,
    failed: tuple[str, ...] = (),
    receipt_ids: tuple[str, ...] = (),
) -> evening_report.EveningNotificationSummary:
    accepted = channels or ("serverchan",)
    return evening_report.EveningNotificationSummary(
        configured_channels=tuple(dict.fromkeys((*accepted, *failed))),
        accepted_channels=accepted,
        failed_channels=failed,
        provider_receipt_ids=receipt_ids,
    )


def _recommendation_repository(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )


def _rejected_summary(*channels: str) -> evening_report.EveningNotificationSummary:
    rejected = channels or ("serverchan", "bark")
    return evening_report.EveningNotificationSummary(
        configured_channels=rejected,
        failed_channels=rejected,
    )


def _register_holding_channels(
    repository: SQLiteRepository,
    channels: tuple[str, ...],
    *,
    chart_channels: tuple[str, ...] = (),
    chart_publisher_id: str | None = None,
    legacy_bool: bool | None = None,
) -> ActiveHoldingPortfolio:
    metadata: dict[str, object] = {
        "holding_summary_delivery_channels": list(channels),
        HOLDING_CHART_DELIVERY_CHANNELS_KEY: list(chart_channels),
        HOLDING_CHART_PUBLISHER_ID_KEY: chart_publisher_id,
    }
    if legacy_bool is not None:
        metadata["external_delivery_consent"] = legacy_bool
    return replace_active_holdings(
        repository,
        (
            HoldingPositionInput(
                symbol="600919",
                name="持仓摘要股票",
                entry_date=CUTOFF,
                cost_price=987654.32,
                stock_sleeve_weight=1.0,
                account_weight=0.731,
            ),
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 27, 21, tzinfo=UTC),
        metadata=metadata,
    )


def _holding_review(
    portfolio_id: str,
    holding_version: int,
    *,
    symbol: str = "600919",
    name: str = "持仓摘要股票",
) -> HoldingTreeReviewSummary:
    row = HoldingTreeReviewRow(
        symbol=symbol,
        name=name,
        holding_weeks=4,
        holding_version=holding_version,
        position_key=f"holding:{symbol}:test",
        status=HoldingReviewRowStatus.READY,
        action=HoldingAction.HOLD,
        latest_close=12.50,
        cost_price=987654.32,
        stock_sleeve_weight=1.0,
        account_weight=0.731,
        candidate_stop=11.80,
        previous_stop=11.50,
        effective_stop=11.80,
        stop_raised=True,
        close_below_stop=False,
        source_timeframe="daily",
        evidence_date=CUTOFF,
        slow_direction="up",
        primary_structure="volume_confirmed_breakout",
        daily_execution="confirmed",
        reasons=("no_completed_close_exit_or_reduce_signal",),
    )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.READY,
        portfolio_id=portfolio_id,
        holding_version=holding_version,
        holding_weeks=4,
        reviewed_at=datetime(2026, 8, 27, 21, tzinfo=UTC),
        data_cutoff=CUTOFF,
        rows=(row,),
    )


def _holding_chart_result(
    portfolio: ActiveHoldingPortfolio,
    *,
    payload: bytes = b"\x89PNG\r\n\x1a\nprivate-composite",
) -> HoldingChartBuildResult:
    identity = SimpleNamespace(
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        data_cutoff=CUTOFF,
    )
    metadata = SimpleNamespace(
        review_identity=identity,
        panel_count=8,
        symbols=tuple(position.symbol for position in portfolio.positions),
        width=1_200,
        height=3_600,
        raw_rows_embedded=False,
        sensitive_fields_embedded=False,
    )
    rendered = SimpleNamespace(composite_png=payload, metadata=metadata)
    return HoldingChartBuildResult(
        status=HoldingChartBuildStatus.READY,
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=portfolio.holding_weeks,
        data_cutoff=CUTOFF,
        rendered=rendered,
        archive=SimpleNamespace(composite_path=Path("private-local-only.png")),
    )


def _register_four_holding_channels(
    repository: SQLiteRepository,
    *,
    summary_channels: tuple[str, ...] = ("serverchan", "bark"),
    chart_channels: tuple[str, ...] = ("serverchan",),
) -> ActiveHoldingPortfolio:
    symbols = ("601101", "603012", "603268", "603679")
    return replace_active_holdings(
        repository,
        tuple(
            HoldingPositionInput(
                symbol=symbol,
                name=f"持仓{index}",
                entry_date=CUTOFF,
                stock_sleeve_weight=0.25,
            )
            for index, symbol in enumerate(symbols, start=1)
        ),
        holding_weeks=4,
        effective_at=datetime(2026, 8, 27, 21, tzinfo=UTC),
        metadata={
            "holding_summary_delivery_channels": list(summary_channels),
            HOLDING_CHART_DELIVERY_CHANNELS_KEY: list(chart_channels),
            HOLDING_CHART_PUBLISHER_ID_KEY: "cloudflare_r2",
        },
    )


def _holding_review_for_portfolio(
    portfolio: ActiveHoldingPortfolio,
) -> HoldingTreeReviewSummary:
    rows = tuple(
        HoldingTreeReviewRow(
            symbol=position.symbol,
            name=position.name,
            holding_weeks=4,
            holding_version=portfolio.version,
            position_key=f"holding:{position.symbol}:test",
            status=HoldingReviewRowStatus.READY,
            action=HoldingAction.HOLD,
            latest_close=12.50,
            cost_price=None,
            stock_sleeve_weight=0.25,
            account_weight=None,
            candidate_stop=11.80,
            previous_stop=11.50,
            effective_stop=11.80,
            stop_raised=True,
            close_below_stop=False,
            source_timeframe="daily",
            evidence_date=CUTOFF,
            slow_direction="up",
            primary_structure="volume_confirmed_breakout",
            daily_execution="confirmed",
            reasons=("no_completed_close_exit_or_reduce_signal",),
        )
        for position in portfolio.positions
    )
    return HoldingTreeReviewSummary(
        status=HoldingReviewSummaryStatus.READY,
        portfolio_id=portfolio.id,
        holding_version=portfolio.version,
        holding_weeks=4,
        reviewed_at=datetime(2026, 8, 27, 21, tzinfo=UTC),
        data_cutoff=CUTOFF,
        rows=rows,
    )


class _PrivatePublisher:
    provider_id = "cloudflare_r2"

    def __init__(self) -> None:
        self.published: list[bytes] = []
        self.revoked: list[str] = []

    def publish_png(self, payload: bytes):
        self.published.append(payload)
        return SimpleNamespace(
            provider_id="cloudflare_r2",
            expires_at=datetime.now(UTC) + timedelta(seconds=3_600),
            image_url="https://images.example.com/private.png?signature=opaque",
            revoke_key="holding-charts/0123456789abcdef0123456789abcdef.png",
        )

    def revoke(self, revoke_key: str):
        self.revoked.append(revoke_key)
        return SimpleNamespace(provider_id="cloudflare_r2", revoked=True)

    def close(self) -> None:
        return None


def test_first_provider_acceptance_writes_state_and_second_run_is_noop(tmp_path: Path) -> None:
    messages = []
    builds = []

    first = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **kwargs: builds.append(kwargs) or _digest(),
        _notifier=lambda message: (
            messages.append(message)
            or _accepted_summary(
                "serverchan",
                "bark",
                receipt_ids=("serverchan:0123456789abcdef",),
            )
        ),
    )
    second = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("NOOP must not rebuild")
        ),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("NOOP must not resend")),
    )

    assert first.exit_code == evening_report.EXIT_OK
    assert first.event["status"] == "provider_accepted"
    assert first.event["delivery_confirmed"] is False
    assert first.event["accepted_channels"] == ["serverchan", "bark"]
    assert first.event["plan_for_date"] == "2026-08-28"
    assert second.exit_code == evening_report.EXIT_OK
    assert second.event["status"] == "noop_no_new_trading_day"
    assert len(builds) == len(messages) == 1
    assert messages[0].title == "A股日报｜2026-08-28（周五）计划"
    assert "2026-08-28周五 A股研究计划" in messages[0].body
    assert "数据2026-08-27" in messages[0].body
    assert "六期限重合与差异审计" not in messages[0].body
    assert len(messages[0].body.encode("utf-8")) <= 2_400
    assert messages[0].compact_body is not None
    assert "数据2026-08-27" in messages[0].compact_body
    assert len(messages[0].compact_body.encode("utf-8")) <= 2_400
    state = json.loads(
        (tmp_path / "state" / "evening-digest-state.json").read_text(encoding="utf-8")
    )
    assert set(state) == {
        "accepted_channels",
        "delivery_confirmed",
        "last_provider_accepted_common_cutoff",
        "plan_for_date",
        "provider_accepted_at",
        "provider_receipt_ids",
        "chart_status",
        "chart_reason",
        "chart_attempts",
    }
    assert state["last_provider_accepted_common_cutoff"] == "2026-08-27"
    assert state["plan_for_date"] == "2026-08-28"
    assert state["delivery_confirmed"] is False
    assert state["provider_receipt_ids"] == ["serverchan:0123456789abcdef"]
    log_path = tmp_path / "logs" / "evening-report.jsonl"
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["status"] for entry in entries] == [
        "provider_accepted",
        "noop_no_new_trading_day",
    ]
    assert entries[0]["exit_code"] == evening_report.EXIT_OK
    assert entries[0]["raw_data_exposed"] is False
    assert entries[0]["orders_enabled"] is False
    assert entries[0]["plan_for_date"] == "2026-08-28"
    assert entries[1]["plan_for_date"] == "2026-08-28"
    assert "A股六周期研究日报" not in log_path.read_text(encoding="utf-8")
    assert log_path.stat().st_mode & 0o777 == 0o600
    assert log_path.parent.stat().st_mode & 0o777 == 0o700
    repository = _recommendation_repository(tmp_path)
    reports = repository.list_recommendation_reports()
    assert len(reports) == 1
    assert reports[0]["archive_nature"] == "original"
    assert reports[0]["plan_for_date"] == "2026-08-28"
    delivery = repository.list_recommendation_delivery_events(reports[0]["id"])
    assert {row["channel"] for row in delivery} == {"serverchan", "bark"}
    assert {row["provider_status"] for row in delivery} == {"provider_accepted"}
    assert all(row["detail_json"]["orders_enabled"] is False for row in delivery)


def test_failed_chart_retries_only_image_with_fresh_publication_and_then_deduplicates(tmp_path):
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_four_holding_channels(repository)
    publisher = _PrivatePublisher()
    messages = []
    builds = []

    def chart(*_args, **_kwargs):
        builds.append(True)
        if len(builds) == 1:
            raise RuntimeError("private provider URL must not be logged")
        return _holding_chart_result(portfolio)

    def notifier(message):
        messages.append(message)
        return evening_report.EveningNotificationSummary(
            configured_channels=("serverchan",),
            accepted_channels=("serverchan",),
            image_accepted_channels=("serverchan",) if message.image_url else (),
        )

    options = dict(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review_for_portfolio(portfolio),
        _build_holding_chart_report=chart,
        _holding_chart_publisher=publisher,
        _notifier=notifier,
    )
    first = evening_report.run_evening_digest(**options)
    assert first.event["chart_status"] == "pending"
    assert first.event["chart_reason"] == "chart_generation_or_upload_failed"
    assert "暂缺" in messages[0].body
    second = evening_report.run_evening_digest(**options)
    assert second.event["chart_status"] == "accepted"
    assert second.event["text_already_accepted"] is True
    assert "补图" in messages[1].title
    assert "六期限计划" not in messages[1].body
    assert messages[1].image_url is not None
    assert len(publisher.published) == 1
    third = evening_report.run_evening_digest(**options)
    assert third.event["status"] == "noop_no_new_trading_day"
    assert len(messages) == 2
    state = (tmp_path / "state" / "evening-digest-state.json").read_text()
    assert "signature" not in state


def test_missing_chart_has_four_bounded_attempts_without_duplicate_text(tmp_path):
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_four_holding_channels(repository)
    messages = []
    options = dict(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review_for_portfolio(portfolio),
        _notifier=lambda message: messages.append(message) or _accepted_summary(),
    )
    outcomes = [evening_report.run_evening_digest(**options) for _ in range(5)]
    assert outcomes[3].event["chart_status"] == "exhausted"
    assert outcomes[3].event["chart_attempts"] == 4
    assert outcomes[4].event["status"] == "noop_no_new_trading_day"
    assert len(messages) == 1


def test_image_rejected_after_text_acceptance_is_revoked_and_republished_with_new_signature(
    tmp_path,
):
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_four_holding_channels(repository)

    class Publisher(_PrivatePublisher):
        def publish_png(self, payload):
            receipt = super().publish_png(payload)
            receipt.image_url += f"&attempt={len(self.published)}"
            return receipt

    publisher = Publisher()
    messages = []

    def notifier(message):
        messages.append(message)
        return evening_report.EveningNotificationSummary(
            configured_channels=("serverchan",),
            accepted_channels=("serverchan",),
            image_accepted_channels=("serverchan",) if len(messages) == 2 else (),
        )

    options = dict(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review_for_portfolio(portfolio),
        _build_holding_chart_report=lambda *_args, **_kwargs: _holding_chart_result(portfolio),
        _holding_chart_publisher=publisher,
        _notifier=notifier,
    )
    first = evening_report.run_evening_digest(**options)
    assert first.event["chart_reason"] == "image_provider_not_accepted"
    assert len(publisher.revoked) == 1
    second = evening_report.run_evening_digest(**options)
    assert second.event["chart_status"] == "accepted"
    assert messages[0].image_url != messages[1].image_url
    assert len(publisher.published) == 2
    assert "六期限计划" not in messages[1].body


def test_evening_window_and_plan_date_block_daytime_past_and_non_eve_reports(tmp_path):
    for hour in (12, 22, 23):
        options = _paths(tmp_path / str(hour))
        options["_clock"] = lambda hour=hour: datetime(2026, 8, 27, hour - 8, tzinfo=UTC)
        result = evening_report.run_evening_digest(
            **options,
            decision_date=CUTOFF,
            _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not send")),
        )
        assert result.event["status"] == "noop_outside_evening_window"
    result = evening_report.run_evening_digest(
        **_paths(tmp_path / "past"),
        decision_date=date(2026, 8, 31),
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: (_ for _ in ()).throw(AssertionError("must not send")),
    )
    assert result.event["status"] == "noop_not_next_session_eve"


def test_holding_summary_consent_is_scoped_per_provider_without_payload_crossing(
    tmp_path: Path,
) -> None:
    cases = (
        ("neither", (), False, False),
        ("server_only", ("serverchan",), True, False),
        ("bark_only", ("bark",), False, True),
        ("both", ("serverchan", "bark"), True, True),
    )
    for label, channels, server_allowed, bark_allowed in cases:
        case_root = tmp_path / label
        repository = _recommendation_repository(case_root)
        portfolio = _register_holding_channels(repository, channels)
        seen: dict[str, str] = {}
        review_calls: list[dict[str, object]] = []

        def notifier(message, *, _seen=seen):
            server = _message_for_channel(message, channel_name="serverchan")
            bark = _message_for_channel(message, channel_name="bark")
            assert server is not None
            assert bark is not None
            _seen["serverchan"] = server.body
            _seen["bark"] = bark.body
            return _accepted_summary("serverchan", "bark")

        def build_holding_review(
            *_args,
            _review_calls=review_calls,
            _portfolio=portfolio,
            **kwargs,
        ):
            _review_calls.append(kwargs)
            return _holding_review(_portfolio.id, _portfolio.version)

        outcome = evening_report.run_evening_digest(
            **_paths(case_root),
            decision_date=CUTOFF,
            _latest_cutoff=lambda _root: CUTOFF,
            _next_trading_day=lambda _cutoff: FRIDAY,
            _build_digest=lambda **_kwargs: _digest(),
            _repository=repository,
            _build_holding_review=build_holding_review,
            _notifier=notifier,
        )

        assert outcome.exit_code == evening_report.EXIT_OK
        assert len(review_calls) == (1 if channels else 0)
        assert ("持仓摘要股票(600919)" in seen["serverchan"]) is server_allowed
        assert ("持仓摘要股票(600919)" in seen["bark"]) is bark_allowed
        assert ("当前持仓修枝" in seen["serverchan"]) is server_allowed
        assert ("当前持仓修枝" in seen["bark"]) is bark_allowed
        for body in seen.values():
            assert "987654.32" not in body
            assert "73.1%" not in body
            assert "总金额" not in body
            assert "账户权重" not in body
            assert len(body.encode("utf-8")) <= 2_400


def test_concurrent_holding_replacement_cannot_disclose_new_portfolio_details(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    _register_holding_channels(repository, ("serverchan", "bark"))
    seen: dict[str, str] = {}

    def build_holding_review(*_args, **_kwargs):
        replacement = replace_active_holdings(
            repository,
            (
                HoldingPositionInput(
                    symbol="601919",
                    name="B组合机密持仓",
                    entry_date=CUTOFF,
                    stock_sleeve_weight=1.0,
                ),
            ),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 27, 22, tzinfo=UTC),
            metadata={"holding_summary_delivery_channels": ["serverchan", "bark"]},
        )
        return _holding_review(
            replacement.id,
            replacement.version,
            symbol="601919",
            name="B组合机密持仓",
        )

    def notifier(message):
        for channel in ("serverchan", "bark"):
            rendered = _message_for_channel(message, channel_name=channel)
            assert rendered is not None
            seen[channel] = rendered.body
        return _accepted_summary("serverchan", "bark")

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=build_holding_review,
        _notifier=notifier,
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    for body in seen.values():
        assert "持仓复核不可用｜本次不生成持仓动作" in body
        assert "B组合机密持仓" not in body
        assert "601919" not in body


def test_concurrent_holding_consent_revocation_invalidates_old_review(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    authorized = _register_holding_channels(repository, ("serverchan", "bark"))
    seen: dict[str, str] = {}

    def build_holding_review(*_args, **_kwargs):
        replace_active_holdings(
            repository,
            (
                HoldingPositionInput(
                    symbol="600919",
                    name="持仓摘要股票",
                    entry_date=CUTOFF,
                    stock_sleeve_weight=1.0,
                ),
            ),
            holding_weeks=4,
            effective_at=datetime(2026, 8, 27, 22, tzinfo=UTC),
            metadata={"holding_summary_delivery_channels": []},
        )
        return _holding_review(authorized.id, authorized.version)

    def notifier(message):
        for channel in ("serverchan", "bark"):
            rendered = _message_for_channel(message, channel_name=channel)
            assert rendered is not None
            seen[channel] = rendered.body
        return _accepted_summary("serverchan", "bark")

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=build_holding_review,
        _notifier=notifier,
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    for body in seen.values():
        assert "持仓复核不可用｜本次不生成持仓动作" in body
        assert "持仓摘要股票" not in body
        assert "600919" not in body


def test_authorized_channel_gets_fail_closed_unavailable_copy_when_review_fails(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    _register_holding_channels(repository, ("serverchan",))
    seen: dict[str, str] = {}

    def notifier(message):
        server = _message_for_channel(message, channel_name="serverchan")
        bark = _message_for_channel(message, channel_name="bark")
        assert server is not None
        assert bark is not None
        seen["serverchan"] = server.body
        seen["bark"] = bark.body
        return _accepted_summary("serverchan", "bark")

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private holding review detail")
        ),
        _notifier=notifier,
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert "持仓复核不可用｜本次不生成持仓动作" in seen["serverchan"]
    assert "当前持仓修枝" not in seen["bark"]
    assert "private holding review detail" not in json.dumps(outcome.event)


def test_legacy_boolean_alone_never_discloses_holding_summary(tmp_path: Path) -> None:
    repository = _recommendation_repository(tmp_path)
    _register_holding_channels(repository, (), legacy_bool=True)
    seen: dict[str, str] = {}

    def notifier(message):
        server = _message_for_channel(message, channel_name="serverchan")
        bark = _message_for_channel(message, channel_name="bark")
        assert server is not None
        assert bark is not None
        seen["serverchan"] = server.body
        seen["bark"] = bark.body
        return _accepted_summary("serverchan", "bark")

    review_calls: list[object] = []
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: (
            review_calls.append(object()) or _holding_review("unused", 1)
        ),
        _notifier=notifier,
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert review_calls == []
    for body in seen.values():
        assert "持仓摘要股票" not in body
        assert "当前持仓修枝" not in body


def test_holding_chart_is_not_built_without_provider_scoped_authorization(
    tmp_path: Path,
) -> None:
    chart_calls: list[object] = []

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _build_holding_chart_report=lambda *_args, **_kwargs: chart_calls.append(object()),
        _notifier=lambda _message: _accepted_summary("serverchan", "bark"),
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert chart_calls == []


def test_summary_authorization_alone_never_builds_or_uploads_holding_chart(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_holding_channels(repository, ("serverchan",))
    chart_calls: list[dict[str, object]] = []
    messages = []

    def build_chart(*_args, **kwargs):
        chart_calls.append(kwargs)
        return _holding_chart_result(portfolio)

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review(
            portfolio.id, portfolio.version
        ),
        _build_holding_chart_report=build_chart,
        _notifier=lambda message: messages.append(message) or _accepted_summary("serverchan"),
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert chart_calls == []
    assert len(messages) == 1
    assert messages[0].image_url is None


def test_authorized_r2_chart_is_attached_only_to_serverchan(tmp_path: Path) -> None:
    repository = _recommendation_repository(tmp_path)
    earlier_portfolio = _register_four_holding_channels(repository)
    portfolio = replace_active_holdings(
        repository,
        tuple(
            HoldingPositionInput(
                symbol=position.symbol,
                name=position.name,
                entry_date=position.entry_date,
                stock_sleeve_weight=position.stock_sleeve_weight,
            )
            for position in earlier_portfolio.positions
        ),
        holding_weeks=earlier_portfolio.holding_weeks,
        effective_at=datetime(2026, 8, 28, 1, tzinfo=UTC),
        metadata={
            "holding_summary_delivery_channels": ["serverchan", "bark"],
            HOLDING_CHART_DELIVERY_CHANNELS_KEY: ["serverchan"],
            HOLDING_CHART_PUBLISHER_ID_KEY: "cloudflare_r2",
        },
    )
    publisher = _PrivatePublisher()
    seen: dict[str, object] = {}
    review_calls: list[dict[str, object]] = []
    chart_calls: list[dict[str, object]] = []

    def build_review(*_args, **kwargs):
        review_calls.append(kwargs)
        return _holding_review_for_portfolio(portfolio)

    def build_chart(*_args, **kwargs):
        chart_calls.append(kwargs)
        return _holding_chart_result(portfolio)

    def notifier(message):
        server = _message_for_channel(message, channel_name="serverchan")
        bark = _message_for_channel(message, channel_name="bark")
        assert server is not None
        assert bark is not None
        seen["root_url"] = message.image_url
        seen["server_url"] = server.image_url
        seen["bark_url"] = bark.image_url
        return evening_report.EveningNotificationSummary(
            configured_channels=("serverchan", "bark"),
            accepted_channels=("serverchan", "bark"),
            image_accepted_channels=("serverchan",),
        )

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=build_review,
        _build_holding_chart_report=build_chart,
        _holding_chart_publisher=publisher,
        _notifier=notifier,
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert len(review_calls) == len(chart_calls) == 1
    review_context = review_calls[0]["holding_context"]
    chart_context = chart_calls[0]["holding_context"]
    assert review_context is chart_context
    assert review_context.portfolio_id == portfolio.id
    assert review_context.version == portfolio.version
    assert review_context.version == earlier_portfolio.version + 1
    assert review_calls[0]["reviewed_at"] is chart_calls[0]["reviewed_at"]
    assert review_context.known_at is review_calls[0]["reviewed_at"]
    assert review_calls[0]["decision_date"] == CUTOFF
    assert chart_calls[0]["as_of"] == CUTOFF
    assert len(publisher.published) == 1
    assert publisher.revoked == []
    assert isinstance(seen["root_url"], str)
    assert seen["server_url"] == seen["root_url"]
    assert seen["bark_url"] is None
    assert outcome.event["image_accepted_channels"] == ["serverchan"]
    log_text = (tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8")
    assert "signature=opaque" not in log_text


def test_uploaded_chart_is_revoked_when_serverchan_does_not_accept_the_image(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_four_holding_channels(repository)
    publisher = _PrivatePublisher()

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review_for_portfolio(portfolio),
        _build_holding_chart_report=lambda *_args, **_kwargs: _holding_chart_result(portfolio),
        _holding_chart_publisher=publisher,
        _notifier=lambda _message: evening_report.EveningNotificationSummary(
            configured_channels=("serverchan", "bark"),
            accepted_channels=("bark",),
            failed_channels=("serverchan",),
        ),
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert len(publisher.published) == 1
    assert publisher.revoked == ["holding-charts/0123456789abcdef0123456789abcdef.png"]


def test_chart_is_revoked_if_holding_revision_changes_during_publication(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_four_holding_channels(repository)

    class ReplacingPublisher(_PrivatePublisher):
        def publish_png(self, payload: bytes):
            receipt = super().publish_png(payload)
            replace_active_holdings(
                repository,
                tuple(
                    HoldingPositionInput(
                        symbol=position.symbol,
                        name=position.name,
                        entry_date=position.entry_date,
                        stock_sleeve_weight=position.stock_sleeve_weight,
                    )
                    for position in portfolio.positions
                ),
                holding_weeks=portfolio.holding_weeks,
                effective_at=datetime(2026, 8, 27, 22, tzinfo=UTC),
                metadata={
                    "holding_summary_delivery_channels": ["serverchan", "bark"],
                    HOLDING_CHART_DELIVERY_CHANNELS_KEY: [],
                    HOLDING_CHART_PUBLISHER_ID_KEY: None,
                },
            )
            return receipt

    publisher = ReplacingPublisher()
    messages = []
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review_for_portfolio(portfolio),
        _build_holding_chart_report=lambda *_args, **_kwargs: _holding_chart_result(portfolio),
        _holding_chart_publisher=publisher,
        _notifier=lambda message: (
            messages.append(message) or _accepted_summary("serverchan", "bark")
        ),
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert publisher.revoked == ["holding-charts/0123456789abcdef0123456789abcdef.png"]
    assert messages[0].image_url is None


def test_chart_authorization_stays_text_only_while_external_publisher_is_disabled(
    tmp_path: Path,
) -> None:
    repository = _recommendation_repository(tmp_path)
    portfolio = _register_holding_channels(
        repository,
        ("serverchan",),
        chart_channels=("serverchan",),
        chart_publisher_id="cloudflare_r2",
    )
    chart_calls: list[object] = []
    published: list[object] = []
    seen: dict[str, object] = {}

    def publisher(*_args, **_kwargs):
        published.append(object())
        raise AssertionError("disabled publisher must not be called")

    def notifier(message):
        seen["root_url"] = message.image_url
        server = _message_for_channel(message, channel_name="serverchan")
        bark = _message_for_channel(message, channel_name="bark")
        assert server is not None
        assert bark is not None
        seen["server_url"] = server.image_url
        seen["bark_url"] = bark.image_url
        return _accepted_summary("serverchan", "bark")

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _repository=repository,
        _build_holding_review=lambda *_args, **_kwargs: _holding_review(
            portfolio.id, portfolio.version
        ),
        _build_holding_chart_report=lambda *_args, **_kwargs: (
            chart_calls.append(object()) or _holding_chart_result(portfolio)
        ),
        _holding_chart_publisher=publisher,
        _notifier=notifier,
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert seen == {
        "root_url": None,
        "server_url": None,
        "bark_url": None,
    }
    assert chart_calls == []
    assert published == []


def test_holding_chart_failures_preserve_text_and_never_leak_private_details(
    tmp_path: Path,
) -> None:
    for label, fail_at in (("builder", "builder"), ("publisher", "publisher")):
        case_root = tmp_path / label
        repository = _recommendation_repository(case_root)
        portfolio = _register_holding_channels(
            repository,
            ("serverchan", "bark"),
            chart_channels=("serverchan", "bark"),
            chart_publisher_id="cloudflare_r2",
        )
        secret = f"600919 /private/{label}/chart.png?signature=do-not-log"
        messages = []
        build_calls: list[object] = []
        publisher_calls: list[object] = []

        def build_chart(
            *_args,
            _fail_at=fail_at,
            _secret=secret,
            _portfolio=portfolio,
            _build_calls=build_calls,
            **_kwargs,
        ):
            _build_calls.append(object())
            if _fail_at == "builder":
                raise RuntimeError(_secret)
            return _holding_chart_result(_portfolio)

        def publisher(
            _payload,
            _metadata,
            *,
            _fail_at=fail_at,
            _secret=secret,
            _publisher_calls=publisher_calls,
        ):
            _publisher_calls.append(object())
            if _fail_at == "publisher":
                raise RuntimeError(_secret)
            return None

        outcome = evening_report.run_evening_digest(
            **_paths(case_root),
            decision_date=CUTOFF,
            _latest_cutoff=lambda _root: CUTOFF,
            _next_trading_day=lambda _cutoff: FRIDAY,
            _build_digest=lambda **_kwargs: _digest(),
            _repository=repository,
            _build_holding_review=lambda *_args, _portfolio=portfolio, **_kwargs: _holding_review(
                _portfolio.id, _portfolio.version
            ),
            _build_holding_chart_report=build_chart,
            _holding_chart_publisher=publisher,
            _notifier=lambda message, _messages=messages: (
                _messages.append(message) or _accepted_summary("serverchan", "bark")
            ),
        )

        assert outcome.exit_code == evening_report.EXIT_OK
        assert build_calls == []
        assert publisher_calls == []
        assert messages[0].image_url is None
        assert "持仓摘要股票(600919)" in messages[0].body
        log_text = (case_root / "logs" / "evening-report.jsonl").read_text(encoding="utf-8")
        assert secret not in log_text
        assert "600919" not in log_text
        assert "/private/" not in log_text


def test_disabled_chart_pipeline_never_reaches_stale_builder_or_publisher(
    tmp_path: Path,
) -> None:
    for label in ("replacement", "revocation"):
        case_root = tmp_path / label
        repository = _recommendation_repository(case_root)
        portfolio = _register_holding_channels(
            repository,
            ("serverchan", "bark"),
            chart_channels=("serverchan", "bark"),
            chart_publisher_id="cloudflare_r2",
        )
        builder_calls: list[object] = []
        publisher_calls: list[object] = []
        messages = []

        def build_chart(
            *_args,
            _label=label,
            _repository=repository,
            _portfolio=portfolio,
            _builder_calls=builder_calls,
            **_kwargs,
        ):
            _builder_calls.append(object())
            replacement_symbol = "601919" if _label == "replacement" else "600919"
            replacement_name = "新组合" if _label == "replacement" else "持仓摘要股票"
            replace_active_holdings(
                _repository,
                (
                    HoldingPositionInput(
                        symbol=replacement_symbol,
                        name=replacement_name,
                        entry_date=CUTOFF,
                        stock_sleeve_weight=1.0,
                    ),
                ),
                holding_weeks=4,
                effective_at=datetime(2026, 8, 27, 22, tzinfo=UTC),
                metadata={
                    "holding_summary_delivery_channels": (
                        ["serverchan", "bark"] if _label == "replacement" else []
                    )
                },
            )
            return _holding_chart_result(_portfolio)

        outcome = evening_report.run_evening_digest(
            **_paths(case_root),
            decision_date=CUTOFF,
            _latest_cutoff=lambda _root: CUTOFF,
            _next_trading_day=lambda _cutoff: FRIDAY,
            _build_digest=lambda **_kwargs: _digest(),
            _repository=repository,
            _build_holding_review=lambda *_args, _portfolio=portfolio, **_kwargs: _holding_review(
                _portfolio.id, _portfolio.version
            ),
            _build_holding_chart_report=build_chart,
            _holding_chart_publisher=lambda *_args, _calls=publisher_calls: (
                _calls.append(object()) or "https://private.example/stale.png?signature=secret"
            ),
            _notifier=lambda message, _messages=messages: (
                _messages.append(message) or _accepted_summary("serverchan", "bark")
            ),
        )

        assert outcome.exit_code == evening_report.EXIT_OK
        assert builder_calls == []
        assert publisher_calls == []
        assert messages[0].image_url is None
        assert "持仓摘要股票" in messages[0].body
        assert "601919" not in messages[0].body


def test_provider_failure_does_not_mark_cutoff_as_accepted(tmp_path: Path) -> None:
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: _rejected_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event["reason"] == "notification_providers_not_accepted"
    assert outcome.event["delivery_confirmed"] is False
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
    reports = _recommendation_repository(tmp_path).list_recommendation_reports()
    assert len(reports) == 1
    delivery = _recommendation_repository(tmp_path).list_recommendation_delivery_events(
        reports[0]["id"]
    )
    assert {row["provider_status"] for row in delivery} == {"provider_failed"}


def test_no_configured_provider_does_not_write_deduplication_state(tmp_path: Path) -> None:
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: evening_report.EveningNotificationSummary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event["reason"] == "notification_channels_not_configured"
    assert outcome.event["delivery_confirmed"] is False
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()


def test_official_infoway_calendar_resolves_first_session_and_closes_adapter() -> None:
    observed: dict[str, object] = {}

    class CalendarProvider:
        def __init__(self, credential: str) -> None:
            observed["credential_was_passed"] = credential == "keychain-only-secret"

        def fetch_cn_trading_days(self, start: date, end: date) -> tuple[date, ...]:
            observed["range"] = (start, end)
            return (FRIDAY, date(2026, 8, 31))

        def close(self) -> None:
            observed["closed"] = True

    result = evening_report.resolve_next_infoway_trading_day(
        CUTOFF,
        _api_key_loader=lambda: "keychain-only-secret",
        _provider_factory=CalendarProvider,
    )

    assert result == FRIDAY
    assert observed == {
        "credential_was_passed": True,
        "range": (date(2026, 8, 28), date(2026, 9, 10)),
        "closed": True,
    }


def test_unverified_next_trading_day_fails_closed_before_notification(tmp_path: Path) -> None:
    secret = "calendar-provider-must-not-leak-this"
    notified: list[object] = []

    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: (_ for _ in ()).throw(RuntimeError(secret)),
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda message: notified.append(message) or _accepted_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event == {
        "job": "ashare-evening-digest",
        "status": "error",
        "reason": "next_trading_day_not_verified",
        "common_cutoff": "2026-08-27",
        "plan_for_date": None,
    }
    assert notified == []
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
    log_text = (tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8")
    assert secret not in log_text
    assert json.loads(log_text)["plan_for_date"] is None


def test_default_notifier_attempts_both_providers_and_keeps_independent_results(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ASHARE_EVENING_NOTIFICATION_CHANNELS", "serverchan,bark")
    from ashare_lab.cli import evening_digest

    closed: list[str] = []

    class Channel:
        def __init__(self, name: str, *, fails: bool = False) -> None:
            self.channel_name = name
            self.fails = fails

        def send(self, _message):
            if self.fails:
                raise NotificationDeliveryError("sanitized")
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
                provider_receipt_id="0123456789abcdef",
            )

        def close(self) -> None:
            closed.append(self.channel_name)

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: "bark-key")
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan", fails=True),
    )
    monkeypatch.setattr(
        evening_digest,
        "BarkNotificationChannel",
        lambda _key: Channel("bark"),
    )

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "正文")
    )

    assert summary.configured_channels == ("serverchan", "bark")
    assert summary.accepted_channels == ("bark",)
    assert summary.failed_channels == ("serverchan",)
    assert summary.provider_receipt_ids == ("bark:0123456789abcdef",)
    assert summary.any_accepted is True
    assert closed == ["serverchan", "bark"]


def test_default_notifier_with_no_configured_channel_is_not_accepted(monkeypatch) -> None:
    from ashare_lab.cli import evening_digest

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: None)
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: None)

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "正文")
    )

    assert summary == evening_report.EveningNotificationSummary()
    assert summary.any_accepted is False


def test_notifier_honors_serverchan_only_process_allow_list(monkeypatch) -> None:
    from ashare_lab.cli import evening_digest

    sent: list[str] = []

    class Channel:
        def __init__(self, name: str) -> None:
            self.channel_name = name

        def send(self, _message):
            sent.append(self.channel_name)
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
            )

        def close(self) -> None:
            return None

    monkeypatch.setenv("ASHARE_EVENING_NOTIFICATION_CHANNELS", "serverchan")
    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(
        evening_digest,
        "load_bark_device_key",
        lambda: (_ for _ in ()).throw(
            AssertionError("Bark key must not be read when the channel is disabled")
        ),
    )
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan"),
    )

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "正文")
    )

    assert summary.configured_channels == ("serverchan",)
    assert summary.accepted_channels == ("serverchan",)
    assert summary.failed_channels == ()
    assert sent == ["serverchan"]


def test_default_notifier_keeps_full_serverchan_body_and_uses_compact_bark_body(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ASHARE_EVENING_NOTIFICATION_CHANNELS", "serverchan,bark")
    from ashare_lab.cli import evening_digest

    seen: dict[str, str] = {}

    class Channel:
        def __init__(self, name: str) -> None:
            self.channel_name = name

        def send(self, message):
            seen[self.channel_name] = message.body
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: "bark-key")
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan"),
    )
    monkeypatch.setattr(
        evening_digest,
        "BarkNotificationChannel",
        lambda _key: Channel("bark"),
    )
    full = "完整六周期正文" * 300
    compact = "六周期紧凑行动摘要"

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage(full[:8], full, compact_body=compact)
    )

    assert summary.accepted_channels == ("serverchan", "bark")
    assert seen == {"serverchan": full, "bark": compact}


def test_oversize_body_without_compact_fails_closed_for_bark_only(monkeypatch) -> None:
    monkeypatch.setenv("ASHARE_EVENING_NOTIFICATION_CHANNELS", "serverchan,bark")
    from ashare_lab.cli import evening_digest

    sent: list[str] = []

    class Channel:
        def __init__(self, name: str) -> None:
            self.channel_name = name

        def send(self, _message):
            sent.append(self.channel_name)
            return NotificationReceipt(
                channel=self.channel_name,
                accepted=True,
                provider_status="provider_accepted",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(evening_digest, "load_serverchan_sendkey", lambda: "server-key")
    monkeypatch.setattr(evening_digest, "load_bark_device_key", lambda: "bark-key")
    monkeypatch.setattr(
        evening_digest,
        "ServerChanNotificationChannel",
        lambda _key: Channel("serverchan"),
    )
    monkeypatch.setattr(
        evening_digest,
        "BarkNotificationChannel",
        lambda _key: Channel("bark"),
    )

    summary = evening_report.send_evening_digest(
        evening_digest.NotificationMessage("行动单", "中" * 801)
    )

    assert sent == ["serverchan"]
    assert summary.accepted_channels == ("serverchan",)
    assert summary.failed_channels == ("bark",)


def test_friday_and_saturday_are_hard_noop_before_state_or_data_reads(
    tmp_path: Path, monkeypatch
) -> None:
    from ashare_lab.cli import evening_digest

    paths = _paths(tmp_path)
    state_path = paths["state_root"] / "evening-digest-state.json"
    state_path.parent.mkdir(parents=True)
    original_state = {
        "last_sent_common_cutoff": "2026-08-27",
        "updated_at": "2026-08-27T21:00:00+08:00",
    }
    state_path.write_text(json.dumps(original_state), encoding="utf-8")
    monkeypatch.setattr(
        evening_digest,
        "_read_state",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("weekend NOOP must not read delivery state")
        ),
    )

    for blocked_date in (FRIDAY, SATURDAY):
        outcome = evening_report.run_evening_digest(
            **paths,
            decision_date=blocked_date,
            _latest_cutoff=lambda _root: (_ for _ in ()).throw(
                AssertionError("weekend NOOP must not inspect market data")
            ),
            _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("weekend NOOP must not build")
            ),
            _notifier=lambda _message: (_ for _ in ()).throw(
                AssertionError("weekend NOOP must not send")
            ),
        )

        assert outcome.exit_code == evening_report.EXIT_OK
        assert outcome.event == {
            "job": "ashare-evening-digest",
            "status": "noop_weekend_send_window_closed",
        }
        assert json.loads(state_path.read_text(encoding="utf-8")) == original_state

    entries = [
        json.loads(line)
        for line in (paths["log_root"] / "evening-report.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["status"] for entry in entries] == [
        "noop_weekend_send_window_closed",
        "noop_weekend_send_window_closed",
    ]
    assert all(entry["exit_code"] == evening_report.EXIT_OK for entry in entries)


def test_sunday_with_no_new_cutoff_uses_normal_deduplication(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    state_path = paths["state_root"] / "evening-digest-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "last_sent_common_cutoff": CUTOFF.isoformat(),
                "updated_at": "2026-08-27T21:00:00+08:00",
            }
        ),
        encoding="utf-8",
    )

    outcome = evening_report.run_evening_digest(
        **paths,
        decision_date=SUNDAY,
        _latest_cutoff=lambda _root: CUTOFF,
        _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("same cutoff must not rebuild on Sunday")
        ),
        _notifier=lambda _message: (_ for _ in ()).throw(
            AssertionError("same cutoff must not resend on Sunday")
        ),
    )

    assert outcome.exit_code == evening_report.EXIT_OK
    assert outcome.event["status"] == "noop_no_new_trading_day"


def test_busy_lock_is_logged_without_building_or_sending(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    lock_path = paths["state_root"] / "daily-sync.lock"

    with daily_update_lock(lock_path) as acquired:
        assert acquired is True
        outcome = evening_report.run_evening_digest(
            **paths,
            decision_date=CUTOFF,
            _latest_cutoff=lambda _root: (_ for _ in ()).throw(
                AssertionError("busy run must not inspect data")
            ),
            _build_digest=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("busy run must not build")
            ),
            _notifier=lambda _message: (_ for _ in ()).throw(
                AssertionError("busy run must not send")
            ),
        )

    assert outcome.exit_code == evening_report.EXIT_RETRY
    assert outcome.event["status"] == "already_running"
    logged = json.loads((tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8"))
    assert logged["status"] == "already_running"
    assert logged["reason"] == "daily_data_lock_busy"


def test_unexpected_failure_never_copies_exception_or_secret(tmp_path: Path) -> None:
    secret = "SCT-do-not-print-this-secret"
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _build_digest=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
        _notifier=lambda _message: _accepted_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert secret not in json.dumps(outcome.event, ensure_ascii=False)
    log_text = (tmp_path / "logs" / "evening-report.jsonl").read_text(encoding="utf-8")
    assert secret not in log_text
    log_event = json.loads(log_text)
    assert log_event["status"] == "error"
    assert log_event["reason"] == "unexpected_evening_digest_error"
    assert set(log_event) <= {
        "accepted_channels",
        "common_cutoff",
        "configured_channels",
        "delivery_confirmed",
        "exit_code",
        "failed_channels",
        "job",
        "logged_at",
        "orders_enabled",
        "period_count",
        "provider_receipt_ids",
        "raw_data_exposed",
        "reason",
        "status",
    }


def test_cli_has_stable_module_and_no_credential_arguments() -> None:
    help_text = evening_report.build_parser().format_help().lower()
    assert "sendkey" not in help_text
    assert "token" not in help_text
    assert "--log-root" in help_text
    assert callable(evening_report.main)


def test_private_log_rotates_and_keeps_at_most_five_backups(tmp_path: Path, monkeypatch) -> None:
    from ashare_lab.cli import evening_digest

    monkeypatch.setattr(evening_digest, "_MAX_LOG_BYTES", 300)
    log_path = tmp_path / "logs" / "evening-report.jsonl"
    outcome = evening_report.EveningDigestOutcome(
        evening_report.EXIT_ERROR,
        {
            "job": "ashare-evening-digest",
            "status": "error",
            "reason": "stable_failure_code",
            "unsafe_exception": "SCT-do-not-log-this",
        },
    )

    for _ in range(30):
        evening_digest._write_log_event(log_path, outcome)

    backups = sorted(log_path.parent.glob("evening-report.jsonl.*"))
    assert 1 <= len(backups) <= 5
    for candidate in (log_path, *backups):
        assert candidate.stat().st_mode & 0o777 == 0o600
        assert "SCT-do-not-log-this" not in candidate.read_text(encoding="utf-8")


def test_log_failure_does_not_turn_rejected_delivery_into_success(
    tmp_path: Path, monkeypatch
) -> None:
    from ashare_lab.cli import evening_digest

    monkeypatch.setattr(
        evening_digest,
        "_write_log_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private log unavailable")),
    )
    outcome = evening_report.run_evening_digest(
        **_paths(tmp_path),
        decision_date=CUTOFF,
        _latest_cutoff=lambda _root: CUTOFF,
        _next_trading_day=lambda _cutoff: FRIDAY,
        _build_digest=lambda **_kwargs: _digest(),
        _notifier=lambda _message: _rejected_summary(),
    )

    assert outcome.exit_code == evening_report.EXIT_ERROR
    assert outcome.event["reason"] == "notification_providers_not_accepted"
    assert not (tmp_path / "state" / "evening-digest-state.json").exists()
