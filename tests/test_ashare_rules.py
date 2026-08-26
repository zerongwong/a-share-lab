from datetime import date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from ashare_lab.domain.market_rules import (
    FINANCING_FRACTION,
    MAX_SPECULATIVE_CAPITAL_FRACTION,
    MAX_WATCHLIST_CANDIDATES,
    MAX_WATCHLIST_SINGLE_NAME_FRACTION,
    T1_EXIT_POLICY_ID,
    WATCHLIST_CANCEL_CUTOFF,
    Board,
    calculate_price_band,
    can_cancel_competitive_order,
    can_sell_on_purchase_day,
    daily_price_limit_rate,
    is_closing_auction,
    is_continuous_auction,
    is_post_close_fixed_price_window,
    is_t1_exit_research_window,
    is_unfilled_order_cancel_window,
    is_watchlist_feature_freeze_window,
    is_watchlist_preflight_window,
    is_watchlist_scan_window,
    is_watchlist_signal_window,
)

CN = ZoneInfo("Asia/Shanghai")


def test_1450_is_continuous_and_closing_auction_cannot_be_cancelled() -> None:
    assert is_continuous_auction(time(14, 50))
    assert not is_continuous_auction(time(14, 57))
    assert is_closing_auction(time(14, 57))
    assert not can_cancel_competitive_order(time(14, 57))


def test_watchlist_research_timeline_is_explicit() -> None:
    assert is_watchlist_preflight_window(time(14, 44, 30))
    assert is_watchlist_scan_window(time(14, 45))
    assert is_watchlist_feature_freeze_window(time(14, 49, 55))
    assert is_watchlist_signal_window(time(14, 50, 5))
    assert time(14, 56, 45) == WATCHLIST_CANCEL_CUTOFF
    assert is_unfilled_order_cancel_window(time(14, 56, 45))
    assert not is_unfilled_order_cancel_window(time(14, 57))
    assert is_t1_exit_research_window(time(9, 35))
    assert is_t1_exit_research_window(time(9, 45))
    assert T1_EXIT_POLICY_ID == "t1_0935_0945_preset_exit_research"


def test_current_board_price_limits_and_first_listing_days() -> None:
    current = date(2026, 8, 24)
    assert daily_price_limit_rate(Board.SH_MAIN, current, listed_trading_days=20) == Decimal("0.10")
    assert daily_price_limit_rate(Board.CHINEXT, current, listed_trading_days=20) == Decimal("0.20")
    assert daily_price_limit_rate(Board.STAR, current, listed_trading_days=20) == Decimal("0.20")
    assert daily_price_limit_rate(Board.BSE, current, listed_trading_days=20) == Decimal("0.30")
    assert daily_price_limit_rate(Board.SH_MAIN, current, listed_trading_days=5) is None
    assert daily_price_limit_rate(Board.BSE, current, listed_trading_days=1) is None


def test_main_board_risk_warning_rule_is_date_versioned() -> None:
    assert daily_price_limit_rate(
        Board.SZ_MAIN,
        date(2026, 7, 3),
        listed_trading_days=100,
        is_risk_warning=True,
    ) == Decimal("0.05")
    assert daily_price_limit_rate(
        Board.SZ_MAIN,
        date(2026, 7, 6),
        listed_trading_days=100,
        is_risk_warning=True,
    ) == Decimal("0.10")


def test_price_band_uses_half_up_tick_rounding() -> None:
    upper, lower = calculate_price_band(Decimal("10.05"), Decimal("0.10"))
    assert upper == Decimal("11.06")
    assert lower == Decimal("9.05")


def test_t_plus_one_and_post_close_window_are_explicit() -> None:
    stamp = datetime(2026, 8, 24, 15, 10, tzinfo=CN)
    assert not can_sell_on_purchase_day(Board.SH_MAIN)
    assert is_post_close_fixed_price_window(Board.SH_MAIN, stamp)
    assert not is_post_close_fixed_price_window(
        Board.SH_MAIN,
        datetime(2026, 7, 3, 15, 10, tzinfo=CN),
    )
    assert Decimal("0") == FINANCING_FRACTION
    assert Decimal("0.05") == MAX_SPECULATIVE_CAPITAL_FRACTION
    assert Decimal("0.01") == MAX_WATCHLIST_SINGLE_NAME_FRACTION
    assert MAX_WATCHLIST_CANDIDATES == 5
