"""Bounded public Tencent/Sina quote snapshots, not a tick-feed/SLA.

Only requested A-share identities are read. Timestamps belong to the quotes,
not the HTTP retrieval time. All raw data remain local to the monitor.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from ashare_lab.ports.market_data import normalize_symbol

CN = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class IntradayQuote:
    symbol: str
    source: str
    price: float
    low: float
    high: float
    previous_close: float
    quoted_at: datetime


def provider_code(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    if symbol.startswith("6"):
        return "sh" + symbol
    if symbol.startswith(("00", "30")):
        return "sz" + symbol
    if symbol.startswith(("4", "8", "92")):
        return "bj" + symbol
    raise ValueError("unsupported_a_share_identity")


def parse_quotes(text: str, source: str, symbols: tuple[str, ...]) -> dict[str, IntradayQuote]:
    prefix, separator = ("v_", "~") if source == "tencent" else ("hq_str_", ",")
    if source not in {"tencent", "sina"}:
        raise ValueError("unsupported_quote_source")
    requested = {provider_code(s): s for s in symbols}
    result = {}
    for code, body in re.findall(rf'{prefix}((?:sh|sz|bj)\d{{6}})="([^\"]*)"', text):
        if code not in requested:
            continue
        symbol = requested[code]
        if symbol in result:
            raise ValueError("duplicate_quote_identity")
        parts = body.split(separator)
        try:
            if source == "tencent":
                price, low, high, previous = (float(parts[i]) for i in (3, 34, 33, 4))
                stamp = datetime.strptime(parts[30], "%Y%m%d%H%M%S").replace(tzinfo=CN)
                if parts[2] != symbol:
                    raise ValueError("quote_identity_mismatch")
            else:
                price, low, high, previous = (float(parts[i]) for i in (3, 5, 4, 2))
                stamp = datetime.fromisoformat(parts[30] + "T" + parts[31]).replace(tzinfo=CN)
            if not all(math.isfinite(v) and v > 0 for v in (price, low, high, previous)):
                continue
            if not low <= price <= high:
                continue
            result[symbol] = IntradayQuote(symbol, source, price, low, high, previous, stamp)
        except (ValueError, IndexError):
            continue
    return result


def fetch_intraday_quotes(symbols: tuple[str, ...], *, client=None) -> dict[str, dict]:
    if not symbols or len(symbols) > 5 or len(set(symbols)) != len(symbols):
        raise ValueError("monitor_requires_one_to_five_unique_holdings")
    codes = ",".join(provider_code(s) for s in symbols)
    own = client is None
    client = client or httpx.Client(timeout=4.0, trust_env=False, follow_redirects=False)
    result = {}
    try:
        for source, root in (
            ("tencent", "https://qt.gtimg.cn/q="),
            ("sina", "https://hq.sinajs.cn/list="),
        ):
            try:
                response = client.get(
                    root + codes, headers={"Referer": "https://finance.sina.com.cn/"}
                )
                response.raise_for_status()
                result[source] = parse_quotes(response.content.decode("gb18030"), source, symbols)
            except (httpx.HTTPError, UnicodeError, ValueError):
                result[source] = {}
    finally:
        if own:
            client.close()
    return result


def fresh_quotes(quotes, *, now: datetime) -> tuple[IntradayQuote, ...]:
    return tuple(
        q
        for q in quotes
        if q.quoted_at.date() == now.astimezone(CN).date()
        and -5 <= (now - q.quoted_at).total_seconds() <= 90
    )


def crosscheck(quotes: tuple[IntradayQuote, ...]) -> bool:
    if {q.source for q in quotes} != {"tencent", "sina"} or len(quotes) != 2:
        return False
    a, b = quotes
    return (
        a.symbol == b.symbol
        and abs((a.quoted_at - b.quoted_at).total_seconds()) <= 60
        and abs(a.price - b.price) <= max(0.03, 0.003 * min(a.price, b.price))
        and abs(a.previous_close - b.previous_close) <= 0.011
    )
