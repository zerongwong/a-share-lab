"""Dated A-share market rules used by the 14:50 research watchlist.

The module deliberately keeps exchange mechanics outside the scoring model.  A
model must not learn (or guess) whether a security is sellable on the purchase
day, what its daily price band is, or whether an order can be cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum


class Board(StrEnum):
    SH_MAIN = "sh_main"
    SZ_MAIN = "sz_main"
    STAR = "star"
    CHINEXT = "chinext"
    BSE = "bse"


@dataclass(frozen=True, slots=True)
class BoardRule:
    price_limit_rate: Decimal
    no_limit_listing_days: int
    buy_lot: int
    supports_post_close_fixed_price: bool


BOARD_RULES: dict[Board, BoardRule] = {
    Board.SH_MAIN: BoardRule(Decimal("0.10"), 5, 100, True),
    Board.SZ_MAIN: BoardRule(Decimal("0.10"), 5, 100, True),
    Board.STAR: BoardRule(Decimal("0.20"), 5, 200, True),
    Board.CHINEXT: BoardRule(Decimal("0.20"), 5, 100, True),
    Board.BSE: BoardRule(Decimal("0.30"), 1, 100, False),
}

# From this date, main-board risk-warning shares changed from 5% to 10%.
MAIN_BOARD_RISK_WARNING_ALIGNMENT_DATE = date(2026, 7, 6)

# From this date, post-close fixed-price trading was expanded to all Shanghai
# and Shenzhen A shares.  STAR and ChiNext already supported it earlier.
ALL_A_SHARE_POST_CLOSE_START_DATE = date(2026, 7, 6)
STAR_POST_CLOSE_START_DATE = date(2019, 7, 22)
CHINEXT_POST_CLOSE_START_DATE = date(2020, 8, 24)

MAX_WATCHLIST_CANDIDATES = 5
MAX_SPECULATIVE_CAPITAL_FRACTION = Decimal("0.05")
MAX_WATCHLIST_SINGLE_NAME_FRACTION = Decimal("0.01")
FINANCING_FRACTION = Decimal("0")
MAX_QUOTE_LATENCY_SECONDS = 3.0

MORNING_CONTINUOUS_START = time(9, 30)
MORNING_CONTINUOUS_END = time(11, 30)
AFTERNOON_CONTINUOUS_START = time(13, 0)
AFTERNOON_CONTINUOUS_END = time(14, 57)
CLOSING_AUCTION_START = time(14, 57)
CLOSING_AUCTION_END = time(15, 0)
POST_CLOSE_START = time(15, 5)
POST_CLOSE_END = time(15, 30)
WATCHLIST_PREFLIGHT_START = time(14, 44)
WATCHLIST_PREFLIGHT_END = time(14, 45)
WATCHLIST_SCAN_START = time(14, 45)
WATCHLIST_SCAN_END = time(14, 50)
WATCHLIST_FEATURE_FREEZE_START = time(14, 49, 55)
WATCHLIST_FEATURE_FREEZE_END = time(14, 50)
WATCHLIST_SIGNAL_START = time(14, 50, 5)
WATCHLIST_SIGNAL_END = time(14, 50, 30)
WATCHLIST_CANCEL_CUTOFF = time(14, 56, 45)
T1_EXIT_RESEARCH_START = time(9, 35)
T1_EXIT_RESEARCH_END = time(9, 45)
T1_EXIT_POLICY_ID = "t1_0935_0945_preset_exit_research"


def board_rule(board: Board) -> BoardRule:
    return BOARD_RULES[Board(board)]


def daily_price_limit_rate(
    board: Board,
    trade_date: date,
    *,
    listed_trading_days: int,
    is_risk_warning: bool = False,
    is_relisting_first_day: bool = False,
    is_delisting_first_day: bool = False,
) -> Decimal | None:
    """Return the price-band rate, or ``None`` when there is no daily band."""

    rule = board_rule(board)
    if listed_trading_days <= rule.no_limit_listing_days:
        return None
    if is_relisting_first_day or is_delisting_first_day:
        return None
    if (
        board in {Board.SH_MAIN, Board.SZ_MAIN}
        and is_risk_warning
        and trade_date < MAIN_BOARD_RISK_WARNING_ALIGNMENT_DATE
    ):
        return Decimal("0.05")
    return rule.price_limit_rate


def calculate_price_band(
    previous_close: Decimal,
    rate: Decimal,
    *,
    tick_size: Decimal = Decimal("0.01"),
) -> tuple[Decimal, Decimal]:
    """Calculate exchange-style upper/lower prices using half-up tick rounding."""

    if previous_close <= 0:
        raise ValueError("previous_close must be positive")
    if rate <= 0:
        raise ValueError("rate must be positive")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    def round_to_tick(value: Decimal) -> Decimal:
        ticks = (value / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return ticks * tick_size

    return (
        round_to_tick(previous_close * (Decimal("1") + rate)),
        round_to_tick(previous_close * (Decimal("1") - rate)),
    )


def _clock(value: datetime | time) -> time:
    return value.timetz().replace(tzinfo=None) if isinstance(value, datetime) else value


def is_continuous_auction(value: datetime | time) -> bool:
    current = _clock(value)
    return (
        MORNING_CONTINUOUS_START <= current < MORNING_CONTINUOUS_END
        or AFTERNOON_CONTINUOUS_START <= current < AFTERNOON_CONTINUOUS_END
    )


def is_closing_auction(value: datetime | time) -> bool:
    current = _clock(value)
    return CLOSING_AUCTION_START <= current <= CLOSING_AUCTION_END


def can_cancel_competitive_order(value: datetime | time) -> bool:
    current = _clock(value)
    accepts_competitive_order = (
        time(9, 15) <= current <= time(9, 25)
        or MORNING_CONTINUOUS_START <= current <= MORNING_CONTINUOUS_END
        or AFTERNOON_CONTINUOUS_START <= current <= CLOSING_AUCTION_END
    )
    opening_no_cancel = time(9, 20) <= current <= time(9, 25)
    closing_no_cancel = CLOSING_AUCTION_START <= current <= CLOSING_AUCTION_END
    return accepts_competitive_order and not (opening_no_cancel or closing_no_cancel)


def is_watchlist_preflight_window(value: datetime | time) -> bool:
    current = _clock(value)
    return WATCHLIST_PREFLIGHT_START <= current < WATCHLIST_PREFLIGHT_END


def is_watchlist_scan_window(value: datetime | time) -> bool:
    current = _clock(value)
    return WATCHLIST_SCAN_START <= current <= WATCHLIST_SCAN_END


def is_watchlist_feature_freeze_window(value: datetime | time) -> bool:
    current = _clock(value)
    return WATCHLIST_FEATURE_FREEZE_START <= current < WATCHLIST_FEATURE_FREEZE_END


def is_watchlist_signal_window(value: datetime | time) -> bool:
    current = _clock(value)
    return WATCHLIST_SIGNAL_START <= current <= WATCHLIST_SIGNAL_END


def is_unfilled_order_cancel_window(value: datetime | time) -> bool:
    """The manual-research window for withdrawing unfilled orders before 14:57."""

    current = _clock(value)
    return WATCHLIST_CANCEL_CUTOFF <= current < CLOSING_AUCTION_START


def is_t1_exit_research_window(value: datetime | time) -> bool:
    current = _clock(value)
    return T1_EXIT_RESEARCH_START <= current <= T1_EXIT_RESEARCH_END


def supports_post_close_fixed_price(board: Board, trade_date: date) -> bool:
    if board == Board.STAR:
        return trade_date >= STAR_POST_CLOSE_START_DATE
    if board == Board.CHINEXT:
        return trade_date >= CHINEXT_POST_CLOSE_START_DATE
    if board in {Board.SH_MAIN, Board.SZ_MAIN}:
        return trade_date >= ALL_A_SHARE_POST_CLOSE_START_DATE
    return False


def is_post_close_fixed_price_window(
    board: Board, value: datetime, *, trade_date: date | None = None
) -> bool:
    effective_date = trade_date or value.date()
    current = _clock(value)
    return (
        supports_post_close_fixed_price(board, effective_date)
        and POST_CLOSE_START <= current <= POST_CLOSE_END
    )


def can_sell_on_purchase_day(board: Board) -> bool:
    """Ordinary A shares are not day-tradable; board is explicit for auditability."""

    Board(board)  # validate the value
    return False
