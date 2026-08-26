from dataclasses import replace
from datetime import datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from ashare_lab.domain.market_rules import T1_EXIT_POLICY_ID, Board
from ashare_lab.services.screen_limit_watchlist import (
    LimitStockSignal,
    LimitWatchlistConfig,
    LimitWatchlistSnapshot,
    ProbabilityBundle,
    TraceableScoreInputs,
    WatchlistStatus,
    build_score_breakdown,
    screen_limit_watchlist,
)

CN = ZoneInfo("Asia/Shanghai")
PREFLIGHT = datetime(2026, 8, 24, 14, 44, 30, tzinfo=CN)
FREEZE = datetime(2026, 8, 24, 14, 49, 55, tzinfo=CN)
OUTPUT = datetime(2026, 8, 24, 14, 50, 5, tzinfo=CN)


def valid_probabilities(**changes: object) -> ProbabilityBundle:
    base = ProbabilityBundle(
        fill_today=0.70,
        close_limit_today=0.60,
        hit_limit_t1=0.55,
        close_limit_t1=0.35,
        positive_return_t1=0.65,
        exit_by_t1_1000=0.90,
        loss_5pct_t1=0.10,
        expected_net_return_t1=0.02,
        cvar95_t1=-0.05,
        calibrated=True,
        confidence_intervals_available=True,
        base_rates_available=True,
        cohort_id="sh-main-first-board-normal-regime",
        sample_size=2_000,
    )
    return replace(base, **changes)


def valid_score(**changes: object) -> TraceableScoreInputs:
    base = TraceableScoreInputs(
        market_environment=0.90,
        sector_confirmation=0.90,
        price_volume_structure=0.90,
        limit_behavior=0.90,
        catalyst_quality=0.90,
        execution_quality=0.90,
        next_day_distribution=0.90,
        tail_risk=0.10,
        evidence_ids=(
            "market-snapshot",
            "sector-snapshot",
            "price-volume-snapshot",
            "l2-limit-behavior",
            "official-event-evidence",
            "l2-fill-simulation",
            "oos-return-distribution",
            "oos-tail-risk",
        ),
    )
    return replace(base, **changes)


def valid_signal(symbol: str, theme: str, **changes: object) -> LimitStockSignal:
    base = LimitStockSignal(
        symbol=symbol,
        name=f"candidate-{symbol}",
        board=Board.SH_MAIN,
        theme=theme,
        quote_timestamp=FREEZE - timedelta(seconds=1),
        last_price=Decimal("10.00"),
        official_limit_up_price=Decimal("10.00"),
        data_complete=True,
        level2_complete=True,
        limit_price_verified=True,
        user_has_board_permission=True,
        board_model_ready=True,
        listed_trading_days=1_000,
        is_st=False,
        is_delisting=False,
        is_suspended=False,
        known_next_session_suspension=False,
        has_major_negative_event=False,
        severe_abnormal_trading_risk=False,
        first_five_day_no_limit=False,
        touched_limit=True,
        at_limit=True,
        estimated_fillable=True,
        one_word_limit=False,
        board_open_count=1,
        cancellation_rate=0.20,
        queue_decay_rate=0.20,
        sector_limit_count=3,
        sector_breadth=0.70,
        market_broken_board_rate=0.20,
        median_turnover_20d=Decimal("500000000"),
        turnover_last_5m=Decimal("1000000000"),
        minimum_lot_affordable=True,
        probabilities=valid_probabilities(),
        score_inputs=valid_score(),
    )
    return replace(base, **changes)


def valid_snapshot(*signals: LimitStockSignal, **changes: object) -> LimitWatchlistSnapshot:
    base = LimitWatchlistSnapshot(
        as_of=OUTPUT,
        preflight_completed_at=PREFLIGHT,
        feature_frozen_at=FREEZE,
        market_quote_timestamp=FREEZE - timedelta(seconds=1),
        market_data_latency_seconds=1.0,
        account_equity=Decimal("1000000"),
        signals=tuple(signals),
        data_complete=True,
        licensed_realtime_data=True,
        clock_synchronized=True,
        trading_calendar_verified=True,
        market_halted=False,
        model_ready=True,
        historical_l2_backtest_ready=True,
        out_of_sample_validation_ready=True,
        shadow_run_ready=True,
        probability_calibration_ready=True,
        model_version="walk-forward-2026-08",
        data_version="licensed-l2-2026-08-24",
        rule_version="cn-equities-2026-07-06",
        exit_policy_id=T1_EXIT_POLICY_ID,
    )
    return replace(base, **changes)


def test_default_is_disabled_and_never_allocates_finances_or_schedules() -> None:
    result = screen_limit_watchlist(valid_snapshot(valid_signal("600001", "robotics")))
    assert result.status is WatchlistStatus.DISABLED
    assert result.candidates == ()
    assert result.total_capital_fraction == 0
    assert result.financing_fraction == 0
    assert not result.auto_order_allowed
    assert not result.system_task_created


def test_enabled_screen_returns_at_most_five_with_one_percent_each() -> None:
    signals = tuple(valid_signal(f"60000{index}", f"theme-{index}") for index in range(1, 7))
    result = screen_limit_watchlist(
        valid_snapshot(*signals),
        LimitWatchlistConfig(enabled=True),
    )
    assert result.status is WatchlistStatus.READY
    assert len(result.candidates) == 5
    assert result.total_capital_fraction == Decimal("0.05")
    assert all(item.capital_fraction == Decimal("0.01") for item in result.candidates)
    assert result.financing_fraction == 0
    assert result.cancel_unfilled_by == time(14, 56, 45)
    assert result.orders_irrevocable_from == time(14, 57)
    assert result.t1_exit_research_window == (time(9, 35), time(9, 45))
    assert all(item.research_only and not item.auto_order_allowed for item in result.candidates)


def test_fewer_candidates_leave_cash_and_same_theme_is_limited_to_two() -> None:
    same_theme = tuple(valid_signal(f"60001{index}", "robotics") for index in range(3))
    result = screen_limit_watchlist(
        valid_snapshot(*same_theme),
        LimitWatchlistConfig(enabled=True),
    )
    assert result.status is WatchlistStatus.READY
    assert len(result.candidates) == 2
    assert result.total_capital_fraction == Decimal("0.02")


def test_traceable_fixed_score_and_missing_component_fail_closed() -> None:
    perfect = valid_score(
        market_environment=1.0,
        sector_confirmation=1.0,
        price_volume_structure=1.0,
        limit_behavior=1.0,
        catalyst_quality=1.0,
        execution_quality=1.0,
        next_day_distribution=1.0,
        tail_risk=0.0,
    )
    breakdown = build_score_breakdown(perfect)
    assert breakdown is not None
    assert breakdown.total == 100.0
    assert breakdown.limit_behavior == 20.0

    missing = valid_signal(
        "600001",
        "robotics",
        score_inputs=valid_score(catalyst_quality=None),
    )
    result = screen_limit_watchlist(
        valid_snapshot(missing),
        LimitWatchlistConfig(enabled=True),
    )
    assert result.status is WatchlistStatus.NO_CANDIDATES


def test_stale_or_incomplete_global_data_returns_no_candidates() -> None:
    signal = valid_signal("600001", "robotics")
    stale = valid_snapshot(
        signal,
        market_quote_timestamp=FREEZE - timedelta(seconds=4),
        market_data_latency_seconds=4.0,
    )
    stale_result = screen_limit_watchlist(stale, LimitWatchlistConfig(enabled=True))
    assert stale_result.status is WatchlistStatus.STALE_DATA
    assert stale_result.candidates == ()

    incomplete = valid_snapshot(signal, licensed_realtime_data=False)
    incomplete_result = screen_limit_watchlist(
        incomplete,
        LimitWatchlistConfig(enabled=True),
    )
    assert incomplete_result.status is WatchlistStatus.INCOMPLETE_DATA
    assert incomplete_result.candidates == ()


def test_hard_risk_flags_and_uncalibrated_probabilities_are_filtered() -> None:
    signals = (
        valid_signal("600001", "risk", is_st=True),
        valid_signal("600002", "risk", one_word_limit=True),
        valid_signal(
            "600003",
            "risk",
            probabilities=valid_probabilities(confidence_intervals_available=False),
        ),
        valid_signal("600004", "risk", level2_complete=False),
    )
    result = screen_limit_watchlist(
        valid_snapshot(*signals),
        LimitWatchlistConfig(enabled=True),
    )
    assert result.status is WatchlistStatus.NO_CANDIDATES
    assert result.candidates == ()


def test_schedule_exit_policy_signal_latency_and_capacity_are_enforced() -> None:
    valid = valid_signal("600001", "robotics")
    wrong_schedule = valid_snapshot(
        valid,
        as_of=datetime(2026, 8, 24, 14, 51, tzinfo=CN),
    )
    schedule_result = screen_limit_watchlist(
        wrong_schedule,
        LimitWatchlistConfig(enabled=True),
    )
    assert schedule_result.status is WatchlistStatus.OUTSIDE_SIGNAL_WINDOW

    wrong_exit = valid_snapshot(valid, exit_policy_id="choose_best_price_tomorrow")
    exit_result = screen_limit_watchlist(
        wrong_exit,
        LimitWatchlistConfig(enabled=True),
    )
    assert exit_result.status is WatchlistStatus.MODEL_NOT_READY

    stale_signal = valid_signal(
        "600002",
        "power",
        quote_timestamp=FREEZE - timedelta(seconds=4),
    )
    too_large = valid_signal(
        "600003",
        "shipping",
        turnover_last_5m=Decimal("100000"),
    )
    result = screen_limit_watchlist(
        valid_snapshot(stale_signal, too_large),
        LimitWatchlistConfig(enabled=True),
    )
    assert result.status is WatchlistStatus.NO_CANDIDATES
    assert result.total_capital_fraction == 0
    assert result.financing_fraction == 0
