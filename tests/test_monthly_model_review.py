from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ashare_lab.services.build_monthly_model_review import (
    BenchmarkEvidence,
    ReviewPopulation,
    build_monthly_model_review,
    build_verified_benchmark_evidence,
    load_monthly_review_completion_state,
    mark_monthly_review_completed,
    monthly_review_due,
    render_monthly_model_review_message,
)


def test_verified_benchmark_uses_action_next_open_and_rejects_reference_observation() -> None:
    formal = _bundle("formal", mode="action_simulation")
    observation = _bundle("observation", mode="observation_simulation")
    observation["members"][0]["observation_anchor"] = "archived_reference_price"
    repository = _Repository([formal, observation])
    dates = tuple(pd.bdate_range("2026-08-03", "2026-08-31").date)

    class Store:
        def read_verified_manifest(self, **_kwargs):
            return pd.DataFrame(
                {
                    "trade_date": dates,
                    "previous_trade_date": (date(2026, 7, 31), *dates[:-1]),
                    "adjustment": "none",
                }
            )

        def read_verified_daily(self, session, **kwargs):
            assert kwargs["asset_kind"] == "indices"
            return pd.DataFrame(
                [
                    {
                        "symbol": "000300.SH",
                        "trade_date": session,
                        "open": 80.0 if session == date(2026, 8, 4) else 100.0,
                        "close": 110.0,
                    }
                ]
            )

    evidence = build_verified_benchmark_evidence(
        repository, overlay_store=Store(), as_of=date(2026, 9, 1)
    )
    assert set(evidence) == {"formal"}
    assert evidence["formal"].entry_date == date(2026, 8, 4)
    assert evidence["formal"].entry_price_field == "open"
    assert evidence["formal"].benchmark_return == pytest.approx(110.0 / 80.0 - 1)


def test_monthly_review_never_pools_calendar_and_trading_clock() -> None:
    legacy = _bundle("legacy", mode="action_simulation", stock_return=0.1)
    natural = _bundle("calendar", mode="action_simulation", stock_return=-0.1)
    natural["batch"]["metadata_json"]["holding_clock"] = "calendar"
    review = build_monthly_model_review(
        _Repository([legacy, natural]), review_month=date(2026, 8, 1), as_of=date(2026, 9, 1)
    )
    active = [row for row in review.horizon_reviews if row.mature_batch_count]
    assert len(active) == 2
    assert {row.holding_clock for row in active} == {"calendar", "trading_sessions"}
    assert all(row.mature_batch_count == 1 for row in active)


class _Repository:
    def __init__(self, bundles: list[dict]) -> None:
        self.bundles = bundles

    def list_recommendation_reports(self, *, limit: int = 100):
        return [item["report"] for item in self.bundles][:limit]

    def list_recommendation_batches(self, report_id: str):
        return [item["batch"] for item in self.bundles if item["report"]["id"] == report_id]

    def get_recommendation_batch_result(self, batch_id: str):
        return next(
            (item["result"] for item in self.bundles if item["batch"]["id"] == batch_id),
            None,
        )

    def list_recommendation_members(self, batch_id: str):
        return next(
            (item["members"] for item in self.bundles if item["batch"]["id"] == batch_id),
            [],
        )

    def list_recommendation_member_results(self, batch_id: str):
        return next(
            (item["member_results"] for item in self.bundles if item["batch"]["id"] == batch_id),
            [],
        )


def _bundle(
    batch_id: str,
    *,
    mode: str,
    archive_nature: str = "original",
    delivered: bool = True,
    stock_return: float | None = 0.04,
    account_return: float | None = 0.02,
    simulated_stock_return: float | None = None,
    simulated_account_return: float | None = None,
    maturity: date = date(2026, 8, 31),
    untriggered: bool = False,
) -> dict:
    report_id = f"report-{batch_id}"
    member_id = f"member-{batch_id}"
    return {
        "report": {
            "id": report_id,
            "archive_nature": archive_nature,
            "plan_for_date": "2026-08-03",
            "cycle_label": "中期下行｜短线修复反弹",
        },
        "batch": {
            "id": batch_id,
            "report_id": report_id,
            "horizon_key": "1w",
            "holding_sessions": 5,
            "label": "1周",
            "evaluation_mode": mode,
            "archive_nature": archive_nature,
            "delivery_accepted": delivered,
            "plan_for_date": "2026-08-03",
            "metadata_json": {
                "risk_nature": "risk_qualified",
                "failure_code": None,
            },
        },
        "result": {
            "batch_id": batch_id,
            "status": "resolved",
            "maturity_date": maturity.isoformat(),
            "stock_sleeve_return": stock_return,
            "account_return": account_return,
            "evaluated_at": f"{maturity.isoformat()}T16:00:00+08:00",
            "updated_at": f"{maturity.isoformat()}T16:00:00+08:00",
            "data_cutoff": maturity.isoformat(),
            "details_json": (
                {
                    "simulated_action_stock_sleeve_return": (
                        stock_return if simulated_stock_return is None else simulated_stock_return
                    ),
                    "simulated_action_account_return": (
                        account_return
                        if simulated_account_return is None
                        else simulated_account_return
                    ),
                }
                if mode == "action_simulation"
                else {}
            ),
        },
        "members": [
            {
                "id": member_id,
                "symbol": "600001",
                "primary_structure": "weekly_breakout",
            }
        ],
        "member_results": [
            {
                "member_id": member_id,
                "status": "not_entered" if untriggered else "resolved",
                "realized_return": stock_return,
                "company_action_clear": 1,
                "details_json": {"condition_triggered": not untriggered},
                "evaluated_at": f"{maturity.isoformat()}T16:00:00+08:00",
                "updated_at": f"{maturity.isoformat()}T16:00:00+08:00",
                "data_cutoff": maturity.isoformat(),
            }
        ],
    }


def _horizon(review, population: ReviewPopulation, key: str = "1w"):
    return next(
        item
        for item in review.horizon_reviews
        if item.population is population and item.horizon_key == key
    )


def test_monthly_review_separates_three_populations_and_six_horizons() -> None:
    formal = _bundle(
        "formal",
        mode="action_simulation",
        simulated_stock_return=0.03,
        simulated_account_return=0.01,
    )
    original = _bundle(
        "original-observation",
        mode="observation_simulation",
        stock_return=-0.10,
        account_return=None,
    )
    reconstructed = _bundle(
        "reconstructed",
        mode="reconstructed_observation",
        archive_nature="reconstructed",
        delivered=False,
        stock_return=0.03,
        account_return=None,
    )
    benchmark = BenchmarkEvidence(
        batch_id="formal",
        benchmark_id="000985.CSI",
        plan_for_date=date(2026, 8, 3),
        maturity_date=date(2026, 8, 31),
        data_cutoff=date(2026, 8, 31),
        evaluated_at=date(2026, 8, 31),
        benchmark_return=0.05,
        adjustment="none",
        source="verified-core-index-basket",
        method_version="benchmark-v1",
    )

    review = build_monthly_model_review(
        _Repository([formal, original, reconstructed]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 9, 1),
        benchmark_evidence_by_batch={"formal": benchmark},
    )

    assert len(review.horizon_reviews) == 18
    formal_summary = _horizon(review, ReviewPopulation.FORMAL_ACTION)
    assert formal_summary.mature_batch_count == 1
    assert formal_summary.mean_weighted_portfolio_return == pytest.approx(0.01)
    assert formal_summary.mean_stock_sleeve_return == pytest.approx(0.03)
    assert formal_summary.mean_benchmark_return == pytest.approx(0.05)
    assert formal_summary.mean_relative_return == pytest.approx(-0.02)
    assert [item.batch_id for item in formal_summary.below_benchmark_batches] == ["formal"]
    assert formal_summary.member_count == 1
    assert formal_summary.data_return_member_count == 1
    assert formal_summary.company_action_evidence_member_count == 1
    assert formal_summary.company_action_clear_member_count == 1
    assert "industry_concentration=unavailable:not_archived_point_in_time" in (
        formal_summary.below_benchmark_batches[0].diagnostic_evidence
    )
    assert "return_basis=conditional_next_open_simulation_untriggered_cash" in (
        formal_summary.below_benchmark_batches[0].diagnostic_evidence
    )

    observation_summary = _horizon(review, ReviewPopulation.ORIGINAL_OBSERVATION)
    assert observation_summary.mean_weighted_portfolio_return == pytest.approx(-0.10)
    assert [item.batch_id for item in observation_summary.negative_batches] == [
        "original-observation"
    ]
    reconstructed_summary = _horizon(review, ReviewPopulation.RECONSTRUCTED_OBSERVATION)
    assert reconstructed_summary.mean_weighted_portfolio_return == pytest.approx(0.03)
    assert review.experiment_proposals == ()
    assert "样本不足" in review.conclusion


def test_formal_action_requires_original_archive_and_accepted_delivery() -> None:
    not_delivered = _bundle(
        "not-delivered",
        mode="action_simulation",
        delivered=False,
    )

    review = build_monthly_model_review(
        _Repository([not_delivered]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 8, 31),
    )

    assert sum(item.mature_batch_count for item in review.horizon_reviews) == 0
    assert review.excluded_batches == ("not-delivered:not_in_three_review_populations",)


def test_formal_no_entry_batch_is_zero_return_cash_not_missing_sample() -> None:
    formal = _bundle(
        "formal-no-entry",
        mode="action_simulation",
        stock_return=0.12,
        account_return=0.06,
        untriggered=True,
    )
    formal["result"]["status"] = "no_entries"
    formal["result"]["details_json"] = {}

    review = build_monthly_model_review(
        _Repository([formal]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 9, 1),
    )

    summary = _horizon(review, ReviewPopulation.FORMAL_ACTION)
    assert summary.valid_return_count == 1
    assert summary.mean_weighted_portfolio_return == pytest.approx(0.0)
    assert summary.mean_stock_sleeve_return == pytest.approx(0.0)


def test_benchmark_must_match_exact_batch_interval_or_remains_unavailable() -> None:
    formal = _bundle("formal", mode="action_simulation")
    mismatched = BenchmarkEvidence(
        batch_id="formal",
        benchmark_id="000985.CSI",
        plan_for_date=date(2026, 8, 4),
        maturity_date=date(2026, 8, 31),
        data_cutoff=date(2026, 8, 31),
        evaluated_at=date(2026, 8, 31),
        benchmark_return=0.01,
        adjustment="none",
        source="verified-index",
        method_version="benchmark-v1",
    )

    review = build_monthly_model_review(
        _Repository([formal]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 8, 31),
        benchmark_evidence_by_batch={"formal": mismatched},
    )

    summary = _horizon(review, ReviewPopulation.FORMAL_ACTION)
    assert summary.benchmark_available_count == 0
    assert summary.mean_relative_return is None
    assert "benchmark_evidence_invalid:formal" in review.evidence_gaps


def test_directional_experiments_require_minimum_sample_and_stay_proposal_only() -> None:
    bundles = [
        _bundle(
            f"formal-{index}",
            mode="action_simulation",
            stock_return=-0.02,
            account_return=-0.01,
            untriggered=True,
        )
        for index in range(12)
    ]
    benchmark = {
        item["batch"]["id"]: BenchmarkEvidence(
            batch_id=item["batch"]["id"],
            benchmark_id="000985.CSI",
            plan_for_date=date(2026, 8, 3),
            maturity_date=date(2026, 8, 31),
            data_cutoff=date(2026, 8, 31),
            evaluated_at=date(2026, 8, 31),
            benchmark_return=0.01,
            adjustment="none",
            source="verified-index",
            method_version="benchmark-v1",
        )
        for item in bundles
    }

    review = build_monthly_model_review(
        _Repository(bundles),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 8, 31),
        benchmark_evidence_by_batch=benchmark,
    )

    proposal_ids = {item.proposal_id for item in review.experiment_proposals}
    assert "2026-08-formal_action-1w-relative-strength-ablation-v1" in proposal_ids
    assert "2026-08-formal_action-1w-entry-confirmation-ablation-v1" in proposal_ids
    assert all(
        item.status == "proposal_only_user_confirmation_required"
        for item in review.experiment_proposals
    )
    assert all(
        "walk-forward" in item.validation_plan or "未见样本" in item.validation_plan
        for item in review.experiment_proposals
    )


def test_incomplete_calendar_month_fails_closed() -> None:
    with pytest.raises(ValueError, match="completed calendar month"):
        build_monthly_model_review(
            _Repository([]),
            review_month=date(2026, 8, 1),
            as_of=date(2026, 8, 30),
        )


def test_result_observed_after_as_of_is_not_used() -> None:
    future_result = _bundle("future-result", mode="action_simulation")
    future_result["result"]["evaluated_at"] = "2026-09-02T08:00:00Z"

    review = build_monthly_model_review(
        _Repository([future_result]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 9, 1),
    )

    assert sum(item.mature_batch_count for item in review.horizon_reviews) == 0
    assert review.excluded_batches == ("future-result:result_not_known_as_of",)


def test_future_member_update_is_excluded_from_historical_diagnostics() -> None:
    bundle = _bundle("future-member", mode="action_simulation", untriggered=True)
    bundle["member_results"][0]["evaluated_at"] = "2026-09-02T08:00:00Z"
    bundle["member_results"][0]["updated_at"] = "2026-09-02T08:00:00Z"

    review = build_monthly_model_review(
        _Repository([bundle]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 9, 1),
    )

    summary = _horizon(review, ReviewPopulation.FORMAL_ACTION)
    assert summary.mature_batch_count == 1
    assert summary.data_return_member_count == 0
    assert summary.untriggered_member_count == 0
    assert "member_result_not_known_as_of_excluded" in review.evidence_gaps


def test_truncated_archive_never_generates_directional_experiment() -> None:
    bundles = [
        _bundle(
            f"formal-{index}",
            mode="action_simulation",
            stock_return=-0.02,
            account_return=-0.01,
            untriggered=True,
        )
        for index in range(12)
    ]

    review = build_monthly_model_review(
        _Repository(bundles),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 8, 31),
        report_limit=12,
    )

    assert review.archive_scan_truncated is True
    assert review.experiment_proposals == ()
    assert "档案扫描已达上限" in review.conclusion
    assert "archive_scan_truncated_no_directional_conclusion" in review.evidence_gaps


def test_benchmark_requires_auditable_cutoff_and_knowledge_time() -> None:
    formal = _bundle("formal", mode="action_simulation")
    future_benchmark = BenchmarkEvidence(
        batch_id="formal",
        benchmark_id="000985.CSI",
        plan_for_date=date(2026, 8, 3),
        maturity_date=date(2026, 8, 31),
        data_cutoff=date(2026, 8, 31),
        evaluated_at=date(2026, 9, 2),
        benchmark_return=0.01,
        adjustment="none",
        source="verified-index",
        method_version="benchmark-v1",
    )

    review = build_monthly_model_review(
        _Repository([formal]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 9, 1),
        benchmark_evidence_by_batch={"formal": future_benchmark},
    )

    summary = _horizon(review, ReviewPopulation.FORMAL_ACTION)
    assert summary.benchmark_available_count == 0
    assert summary.mean_relative_return is None
    assert "benchmark_evidence_invalid:formal" in review.evidence_gaps


def test_message_is_concise_and_does_not_send_or_claim_model_change() -> None:
    review = build_monthly_model_review(
        _Repository([_bundle("formal", mode="action_simulation")]),
        review_month=date(2026, 8, 1),
        as_of=date(2026, 8, 31),
    )

    message = render_monthly_model_review_message(review)

    assert message.title == "A股模型月度复盘｜2026-08"
    assert "六期限重合与差异审计" not in message.body
    assert "不会自动改参数" in message.body
    assert "当前自动流程未接入同区间点时点基准" in message.body
    assert len(message.compact_body.encode("utf-8")) < 2400


def test_monthly_due_waits_for_first_verified_session_and_deduplicates() -> None:
    # September 1 may be a holiday: an August review is not due while the
    # latest verified session is still in August.
    assert (
        monthly_review_due(
            as_of=date(2026, 9, 1),
            latest_verified_session=date(2026, 8, 31),
        )
        is None
    )
    assert monthly_review_due(
        as_of=date(2026, 9, 2),
        latest_verified_session=date(2026, 9, 2),
    ) == date(2026, 8, 1)
    assert (
        monthly_review_due(
            as_of=date(2026, 9, 2),
            latest_verified_session=date(2026, 9, 2),
            completed_months=("2026-08",),
        )
        is None
    )


def test_monthly_completion_state_is_atomic_idempotent_and_secret_free(tmp_path) -> None:
    state_path = tmp_path / "monthly-review-state.json"

    assert load_monthly_review_completion_state(state_path).completed_months == ()
    first = mark_monthly_review_completed(state_path, review_month="2026-08")
    second = mark_monthly_review_completed(state_path, review_month="2026-08")

    assert first.completed_months == ("2026-08",)
    assert second == first
    assert load_monthly_review_completion_state(state_path) == first
    assert "token" not in state_path.read_text(encoding="utf-8").lower()
