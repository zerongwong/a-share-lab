from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from ashare_lab.services.run_recommendation_performance import (
    CorporateActionEvidence,
    render_maturity_notification,
    run_recommendation_performance,
)

PREDECESSOR = date(2026, 8, 27)
SESSIONS = (
    date(2026, 8, 28),
    date(2026, 8, 31),
    date(2026, 9, 1),
    date(2026, 9, 2),
    date(2026, 9, 3),
    date(2026, 9, 4),
)


@dataclass(frozen=True)
class _Dispatch:
    successful_channels: tuple[str, ...]
    failed_channels: tuple[str, ...] = ()


class _Store:
    def __init__(
        self,
        *,
        through: date = SESSIONS[-1],
        action_on: date | None = None,
        missing: tuple[str, date] | None = None,
    ) -> None:
        self.action_on = action_on
        self.frames: dict[date, pd.DataFrame] = {}
        dates = (PREDECESSOR, *SESSIONS)
        previous_close = {"600001": 10.0, "600002": 20.0}
        for trade_date in dates:
            rows = []
            for symbol, initial in (("600001", 10.0), ("600002", 20.0)):
                if missing == (symbol, trade_date):
                    continue
                if trade_date == PREDECESSOR:
                    open_, close = initial, initial
                elif trade_date == SESSIONS[0]:
                    open_, close = (11.0, 11.0) if symbol == "600001" else (20.0, 20.0)
                elif trade_date == SESSIONS[1]:
                    open_, close = (12.0, 12.0) if symbol == "600001" else (20.0, 20.0)
                elif trade_date == SESSIONS[-1]:
                    open_, close = (15.0, 15.0) if symbol == "600001" else (22.0, 22.0)
                else:
                    open_, close = previous_close[symbol], previous_close[symbol]
                prev_close = previous_close[symbol]
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "open": open_,
                        "high": max(open_, close) + 0.2,
                        "low": min(open_, close) - 0.2,
                        "close": close,
                        "prev_close": prev_close,
                        "volume_shares": 1_000.0,
                        "amount_cny": 10_000.0,
                    }
                )
                previous_close[symbol] = close
            self.frames[trade_date] = pd.DataFrame(rows)
        self._through = through

    def read_verified_manifest(self, *, source_id: str) -> pd.DataFrame:
        assert source_id == "zero_budget_eod"
        rows = []
        previous = date(2026, 8, 26)
        for trade_date in (PREDECESSOR, *SESSIONS):
            if trade_date > self._through:
                break
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "previous_trade_date": pd.Timestamp(previous),
                    "adjustment": "none",
                }
            )
            previous = trade_date
        return pd.DataFrame(rows)

    def read_verified_daily(
        self, trade_date: date, *, source_id: str, asset_kind: str
    ) -> pd.DataFrame:
        assert source_id == "zero_budget_eod"
        assert asset_kind == "stocks"
        return self.frames[trade_date].copy()

    def load_corporate_actions(
        self,
        *,
        symbols: tuple[str, ...],
        start: date,
        end: date,
    ) -> CorporateActionEvidence:
        actions = {}
        if self.action_on is not None and start <= self.action_on <= end:
            actions["600001"] = frozenset({self.action_on})
        return CorporateActionEvidence(
            coverage_symbols=frozenset(symbols),
            action_dates_by_symbol=actions,
        )


class _Repository:
    def __init__(self, *, mode: str = "action") -> None:
        reconstructed = mode == "reconstructed"
        observation = mode in {"observation", "reconstructed"}
        self.batch = {
            "id": "batch-1w",
            "report_id": "report-1",
            "decision_date": "2026-08-27",
            "common_cutoff": "2026-08-27",
            "plan_for_date": "2026-08-28",
            "archive_nature": "reconstructed" if reconstructed else "original",
            "delivery_accepted": not reconstructed,
            "horizon_key": "1w",
            "label": "1周",
            "holding_weeks": 1,
            "holding_sessions": 5,
            "evaluation_mode": (
                "reconstructed_observation"
                if reconstructed
                else "observation_simulation"
                if observation
                else "action_simulation"
            ),
            "cohort_nature": (
                "observation_only"
                if reconstructed
                else "risk_qualified"
                if observation
                else "action_qualified"
            ),
            "stock_exposure": 0.5,
        }
        self.members = [
            {
                "id": "member-1",
                "batch_id": "batch-1w",
                "symbol": "600001",
                "name": "样本一",
                "stock_sleeve_weight": 1.0,
                "account_weight": 0.5,
                "reference_price": 10.0,
                "observation_anchor": ("archived_reference_price" if observation else None),
                "entry_plan_json": (
                    None
                    if observation
                    else {
                        "kind": "reclaim_close_confirmation",
                        "trigger_price": 10.0,
                        "invalidation_price": 8.0,
                    }
                ),
            }
        ]
        self.member_results: dict[str, dict[str, Any]] = {}
        self.batch_results: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def list_recommendation_batches_pending_settlement(self, *, as_of=None):
        result = self.batch_results.get("batch-1w")
        if result and result["status"] not in {
            "pending",
            "partial",
            "needs_review",
            "data_quality_failure",
        }:
            return []
        return [dict(self.batch)]

    def list_recommendation_members(self, batch_id):
        assert batch_id == "batch-1w"
        return [dict(value) for value in self.members]

    def record_recommendation_member_result(self, result):
        self.member_results[str(result["member_id"])] = dict(result)

    def record_recommendation_batch_result(self, result):
        self.batch_results[str(result["batch_id"])] = dict(result)

    def list_maturity_results_pending_notification(self, *, limit=100):
        result = self.batch_results.get("batch-1w")
        accepted = any(
            event["delivery_kind"] == "maturity_provider_accepted"
            and event.get("detail_json", {}).get("result_status") == result.get("status")
            and event.get("detail_json", {}).get("result_method_version")
            == result.get("method_version")
            for event in self.events
        )
        if not result or result["status"] == "pending" or accepted:
            return []
        return [{**result, **self.batch, "batch_id": "batch-1w"}]

    def get_recommendation_batch_performance(self, batch_id):
        assert batch_id == "batch-1w"
        return {
            "batch": self.batch,
            "result": self.batch_results.get(batch_id),
            "members": [
                {
                    "recommendation": self.members[0],
                    "result": self.member_results.get("member-1"),
                }
            ],
        }

    def record_recommendation_delivery_event(self, event):
        if not any(value["id"] == event["id"] for value in self.events):
            self.events.append(dict(event))


def _clock() -> datetime:
    return datetime(2026, 9, 4, 11, tzinfo=UTC)


def _run(
    *,
    repository: _Repository,
    store: _Store | None = None,
    notifier=None,
):
    selected_store = store or _Store()
    return run_recommendation_performance(
        repository=repository,
        overlay_store=selected_store,
        notifier=notifier,
        clock=_clock,
        corporate_action_loader=selected_store.load_corporate_actions,
    )


def test_pending_batch_is_persisted_without_notification() -> None:
    repository = _Repository()
    messages = []

    summary = _run(
        repository=repository,
        store=_Store(through=date(2026, 9, 2)),
        notifier=lambda message: messages.append(message),
    )

    assert summary.persisted_batches == 1
    assert repository.batch_results["batch-1w"]["status"] == "pending"
    assert summary.notification_attempts == 0
    assert messages == []


def test_formal_action_reports_published_and_next_open_returns() -> None:
    repository = _Repository()
    sent = []

    def notify(message):
        sent.append(message)
        return _Dispatch(("serverchan",))

    summary = _run(
        repository=repository,
        notifier=notify,
    )

    member = repository.member_results["member-1"]
    batch = repository.batch_results["batch-1w"]
    assert member["realized_return"] == pytest.approx(0.5)
    assert member["details_json"]["simulated_action_return"] == pytest.approx(0.25)
    assert batch["stock_sleeve_return"] == pytest.approx(0.5)
    assert batch["account_return"] == pytest.approx(0.25)
    assert batch["details_json"]["simulated_action_account_return"] == pytest.approx(0.125)
    assert summary.notification_accepted_batches == 1
    assert "发布参考价→到期收盘" in sent[0].body
    assert "次交易日开盘模拟价12.00" in sent[0].body


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("observation", "原始观察组合"),
        ("reconstructed", "历史重建观察组合"),
    ],
)
def test_observation_populations_are_labelled_separately(mode: str, expected: str) -> None:
    repository = _Repository(mode=mode)
    sent = []

    _run(
        repository=repository,
        notifier=lambda message: (sent.append(message), _Dispatch(("bark",)))[1],
    )

    assert expected in sent[0].body
    assert "不并入正式行动业绩" in sent[0].body


def test_corporate_action_break_is_flagged_and_never_counted_as_return() -> None:
    repository = _Repository(mode="observation")

    _run(
        repository=repository,
        store=_Store(action_on=SESSIONS[2]),
    )

    member = repository.member_results["member-1"]
    assert member["status"] == "corporate_action_detected"
    assert member["realized_return"] is None
    assert repository.batch_results["batch-1w"]["status"] == "data_quality_failure"


def test_ohlcv_without_independent_corporate_action_source_never_claims_clear() -> None:
    repository = _Repository(mode="observation")

    run_recommendation_performance(
        repository=repository,
        overlay_store=_Store(),
        clock=_clock,
    )

    member = repository.member_results["member-1"]
    assert member["status"] == "corporate_action_evidence_unknown"
    assert member["company_action_clear"] is None
    assert member["realized_return"] is None
    assert member["details_json"]["raw_unadjusted_price_change"] == pytest.approx(0.5)
    assert repository.batch_results["batch-1w"]["status"] == "data_quality_failure"
    assert repository.batch_results["batch-1w"]["details_json"][
        "raw_unadjusted_stock_sleeve_change"
    ] == pytest.approx(0.5)

    performance = repository.get_recommendation_batch_performance("batch-1w")
    assert performance is not None
    message = render_maturity_notification(performance)
    assert "原始变化+50.00%" in message.body
    assert "公司行动证据不足，不计正式收益" in message.body
    assert "不计正式收益或胜率" in message.body


def test_missing_maturity_price_is_not_silently_zero() -> None:
    repository = _Repository(mode="observation")

    _run(
        repository=repository,
        store=_Store(missing=("600001", SESSIONS[-1])),
    )

    member = repository.member_results["member-1"]
    assert member["status"] == "maturity_price_missing"
    assert member["realized_return"] is None


def test_settlement_and_provider_acceptance_are_idempotent() -> None:
    repository = _Repository()
    messages = []

    def notifier(message):
        messages.append(message)
        return _Dispatch(("serverchan",))

    first = _run(
        repository=repository,
        notifier=notifier,
    )
    second = _run(
        repository=repository,
        notifier=notifier,
    )

    assert first.notification_attempts == 1
    assert second.notification_attempts == 0
    assert len(messages) == 1
    assert len(repository.events) == 1


def test_needs_review_acceptance_does_not_suppress_later_final_result() -> None:
    repository = _Repository(mode="observation")
    messages = []

    def notifier(message):
        messages.append(message)
        return _Dispatch(("serverchan",))

    first = run_recommendation_performance(
        repository=repository,
        overlay_store=_Store(),
        notifier=notifier,
        clock=_clock,
    )
    second = _run(repository=repository, notifier=notifier)
    third = _run(repository=repository, notifier=notifier)

    assert first.notification_accepted_batches == 1
    assert second.notification_accepted_batches == 1
    assert third.notification_attempts == 0
    assert len(messages) == 2
    assert len(repository.events) == 2
    assert {event["detail_json"]["result_status"] for event in repository.events} == {
        "data_quality_failure",
        "settled",
    }


def test_failed_notification_remains_retryable() -> None:
    repository = _Repository()

    first = _run(
        repository=repository,
        notifier=lambda _message: _Dispatch((), ("serverchan",)),
    )
    second = _run(
        repository=repository,
        notifier=lambda _message: _Dispatch(("serverchan",)),
    )

    assert first.notification_failed_batches == 1
    assert first.notification_accepted_batches == 0
    assert second.notification_accepted_batches == 1
    assert len(repository.events) == 1


def test_renderer_does_not_renormalize_or_hide_cash() -> None:
    repository = _Repository()
    _run(
        repository=repository,
    )

    message = render_maturity_notification(
        repository.get_recommendation_batch_performance("batch-1w")
    )

    assert "原权重、不重标" in message.body
    assert "股票仓收益" in message.body
    assert "总资金收益" in message.body
    assert "未入场现金" in message.body
