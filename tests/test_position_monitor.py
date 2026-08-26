from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from ashare_lab.domain.market_rules import Board
from ashare_lab.services.monitor_position_ranges import (
    BrokerPositionInventory,
    IntradayRangePlan,
    PositionMonitorConfig,
    PositionMonitorSnapshot,
    PositionMonitorStatus,
    RangeAlertKind,
    RangeProbabilityBundle,
    RealtimeQuote,
    monitor_position_ranges,
)

CN = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 24, 14, 30, tzinfo=CN)


def valid_probabilities(**changes: object) -> RangeProbabilityBundle:
    base = RangeProbabilityBundle(
        low_fill_within_60s=0.80,
        high_sell_fill_within_60s=0.85,
        low_to_high_before_invalidation=0.55,
        high_to_low_before_close=0.35,
        invalidation_before_high=0.20,
        calibrated=True,
        confidence_intervals_available=True,
        base_rates_available=True,
        out_of_sample_validated=True,
        shadow_run_validated=True,
        cohort_id="sh-main-normal-liquidity-atr-bucket-3",
        model_version="range-model-1",
        calibration_version="isotonic-2026-08",
        sample_size=2_000,
    )
    return replace(base, **changes)


def valid_plan(symbol: str = "600001", **changes: object) -> IntradayRangePlan:
    base = IntradayRangePlan(
        symbol=symbol,
        valid_trade_date=date(2026, 8, 24),
        generated_at=NOW - timedelta(minutes=30),
        evidence_cutoff=NOW - timedelta(days=3),
        low_zone_low=Decimal("9.50"),
        low_zone_high=Decimal("9.70"),
        high_zone_low=Decimal("10.30"),
        high_zone_high=Decimal("10.50"),
        invalidation_below=Decimal("9.20"),
        planned_max_quantity=1_000,
        evidence_ids=("daily-levels-2026-08-21", "intraday-plan-v1"),
        data_complete=True,
        no_lookahead_verified=True,
        probabilities=valid_probabilities(),
    )
    return replace(base, **changes)


def valid_quote(symbol: str = "600001", **changes: object) -> RealtimeQuote:
    base = RealtimeQuote(
        symbol=symbol,
        board=Board.SH_MAIN,
        timestamp=NOW - timedelta(seconds=1),
        last_price=Decimal("10.40"),
        best_bid=Decimal("10.39"),
        best_ask=Decimal("10.40"),
        official_limit_down=Decimal("9.00"),
        official_limit_up=Decimal("11.00"),
        data_complete=True,
        trading_state_normal=True,
        is_suspended=False,
    )
    return replace(base, **changes)


def valid_position(symbol: str = "600001", **changes: object) -> BrokerPositionInventory:
    # 1,000 shares were settled before today. 100 were sold, 200 bought today,
    # 100 are frozen by a pending sell, leaving exactly 800 broker-sellable.
    base = BrokerPositionInventory(
        symbol=symbol,
        total_quantity=1_100,
        settled_before_today_quantity=1_000,
        sellable_quantity=800,
        today_bought_quantity=200,
        today_sold_quantity=100,
        frozen_sell_quantity=100,
    )
    return replace(base, **changes)


def valid_snapshot(**changes: object) -> PositionMonitorSnapshot:
    base = PositionMonitorSnapshot(
        as_of=NOW,
        account_timestamp=NOW - timedelta(seconds=1),
        selected_symbols=("600001",),
        quotes=(valid_quote(),),
        positions=(valid_position(),),
        plans=(valid_plan(),),
        available_cash=Decimal("100000"),
        licensed_realtime_data=True,
        broker_account_authoritative=True,
        broker_account_read_only=True,
        account_data_complete=True,
        clock_synchronized=True,
        trading_calendar_verified=True,
        market_halted=False,
    )
    return replace(base, **changes)


def test_default_is_disabled_and_never_sends_or_executes() -> None:
    result = monitor_position_ranges(valid_snapshot())

    assert result.status is PositionMonitorStatus.DISABLED
    assert result.alerts == ()
    assert result.auto_order_allowed is False
    assert result.notification_sent is False
    assert result.system_task_created is False


def test_high_zone_sell_uses_only_remaining_t_minus_one_sellable_inventory() -> None:
    result = monitor_position_ranges(valid_snapshot(), PositionMonitorConfig(enabled=True))

    assert result.status is PositionMonitorStatus.READY
    alert = result.alerts[0]
    assert alert.kind is RangeAlertKind.HIGH_SELL_RESEARCH
    assert alert.maximum_research_quantity == 800
    assert alert.today_bought_locked_quantity == 200
    assert "今日买入不可当日卖出" in alert.inventory_rule
    assert alert.auto_order_allowed is False


def test_low_zone_buy_back_is_capped_by_quantity_already_sold_today() -> None:
    quote = valid_quote(
        last_price=Decimal("9.60"),
        best_bid=Decimal("9.59"),
        best_ask=Decimal("9.60"),
    )
    snapshot = valid_snapshot(quotes=(quote,))

    result = monitor_position_ranges(snapshot, PositionMonitorConfig(enabled=True))

    assert result.status is PositionMonitorStatus.READY
    assert result.alerts[0].kind is RangeAlertKind.LOW_BUY_RESEARCH
    assert result.alerts[0].maximum_research_quantity == 100
    assert "当日锁定" in result.alerts[0].inventory_rule


def test_today_only_inventory_never_creates_a_sell_or_t_alert() -> None:
    today_only = BrokerPositionInventory(
        symbol="600001",
        total_quantity=500,
        settled_before_today_quantity=0,
        sellable_quantity=0,
        today_bought_quantity=500,
        today_sold_quantity=0,
        frozen_sell_quantity=0,
    )
    snapshot = valid_snapshot(positions=(today_only,))

    result = monitor_position_ranges(snapshot, PositionMonitorConfig(enabled=True))

    assert result.status is PositionMonitorStatus.INCOMPLETE_DATA
    assert result.alerts == ()


def test_stale_or_non_authoritative_inputs_fail_closed_without_partial_alerts() -> None:
    stale = valid_snapshot(
        account_timestamp=NOW - timedelta(seconds=4),
        quotes=(replace(valid_quote(), timestamp=NOW - timedelta(seconds=4)),),
    )
    stale_result = monitor_position_ranges(stale, PositionMonitorConfig(enabled=True))
    assert stale_result.status is PositionMonitorStatus.STALE_DATA
    assert stale_result.alerts == ()

    untrusted = valid_snapshot(broker_account_authoritative=False)
    untrusted_result = monitor_position_ranges(untrusted, PositionMonitorConfig(enabled=True))
    assert untrusted_result.status is PositionMonitorStatus.INCOMPLETE_DATA
    assert untrusted_result.alerts == ()


def test_missing_probability_calibration_or_matching_symbol_fails_closed() -> None:
    uncalibrated = valid_plan(probabilities=RangeProbabilityBundle())
    model_result = monitor_position_ranges(
        valid_snapshot(plans=(uncalibrated,)),
        PositionMonitorConfig(enabled=True),
    )
    assert model_result.status is PositionMonitorStatus.MODEL_NOT_READY
    assert model_result.alerts == ()

    shape_result = monitor_position_ranges(
        valid_snapshot(selected_symbols=("600001", "000001")),
        PositionMonitorConfig(enabled=True),
    )
    assert shape_result.status is PositionMonitorStatus.INCOMPLETE_DATA
    assert shape_result.alerts == ()


def test_outside_zone_returns_zero_alerts_as_a_normal_result() -> None:
    quote = valid_quote(
        last_price=Decimal("10.00"),
        best_bid=Decimal("9.99"),
        best_ask=Decimal("10.00"),
    )

    result = monitor_position_ranges(
        valid_snapshot(quotes=(quote,)),
        PositionMonitorConfig(enabled=True),
    )

    assert result.status is PositionMonitorStatus.NO_ALERTS
    assert result.alerts == ()
