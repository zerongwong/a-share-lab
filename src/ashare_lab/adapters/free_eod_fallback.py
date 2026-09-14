"""Explicit free metadata/index failover. Never substitute on a quality failure.

Individual-stock source remains Tushare with independent AKShare verification:
historical web bars lacking exchange pre-close are not safe substitutes for it.
Index backup is Eastmoney unadjusted OHLCV/amount, checked against Tencent for
ALL six core indices. Index pre-close uses the immediately preceding verified
calendar session; this must never be reused as a stock ex-right reference.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from ashare_lab.adapters.baostock_eod import BAOSTOCK_CORE_INDEX_SYMBOLS
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import DailyIncrementBatch
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS


class FreeEodMetadataFallback:
    def __init__(self, primary, metadata_backup, *, clock=None, client=None):
        self.primary, self.backup = primary, metadata_backup
        self.clock = clock or (lambda: datetime.now(UTC))
        self.client = client or httpx.Client(timeout=10.0, follow_redirects=False)
        self.owns_client = client is None
        self.metadata_sources = {}

    def close(self):
        if self.owns_client:
            self.client.close()

    def _metadata(self, method, *args):
        try:
            result = getattr(self.primary, method)(*args)
            source = "baostock"
        except DataUnavailableError:
            result = getattr(self.backup, method)(*args)
            source = "tushare"
        self.metadata_sources[method] = source
        return result

    def fetch_cn_trading_days(self, start, end):
        return self._metadata("fetch_cn_trading_days", start, end)

    def fetch_cn_stock_symbols(self):
        return self._metadata("fetch_cn_stock_symbols")

    def fetch_core_index_daily(self, target_date, *, cutoff_timestamp=None):
        try:
            return self.primary.fetch_core_index_daily(
                target_date, cutoff_timestamp=cutoff_timestamp
            )
        except DataUnavailableError:
            return self._backup_indices(target_date, cutoff_timestamp=cutoff_timestamp)

    def _json(self, url, params):
        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            raise DataUnavailableError("免费指数备用网络不可用。") from None
        except ValueError:
            raise DataQualityError("免费指数备用响应结构改变。") from None

    def _backup_indices(self, target, *, cutoff_timestamp):
        now = self.clock()
        if now.tzinfo is None:
            raise DataQualityError("指数抓取时间缺少时区。")
        from zoneinfo import ZoneInfo

        local = now.astimezone(ZoneInfo("Asia/Shanghai"))
        if target > local.date() or (target == local.date() and local.hour < 15):
            raise DataUnavailableError("指数目标交易日尚未收盘。")
        days = self.fetch_cn_trading_days(target - timedelta(days=45), target)
        if len(days) < 2 or days[-1] != target:
            raise DataUnavailableError("指数备用前一交易日未核定。")
        previous = days[-2].isoformat()
        records = []
        for symbol in BAOSTOCK_CORE_INDEX_SYMBOLS:
            code, exchange = symbol.split(".")
            txcode = exchange.lower() + code
            em = self._json(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                {
                    "secid": ("1." if exchange == "SH" else "0.") + code,
                    "fields1": "f1,f2,f3,f4,f5",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
                    "klt": "101",
                    "fqt": "0",
                    "beg": days[-2].strftime("%Y%m%d"),
                    "end": target.strftime("%Y%m%d"),
                },
            )
            tx = self._json(
                "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
                {
                    "param": f"{txcode},day,{previous},{target.isoformat()},10,",
                },
            )
            try:
                emdata = em["data"]
                if emdata["code"] != code:
                    raise ValueError("identity mismatch")
                erows = [r.split(",") for r in emdata["klines"]]
                trows = tx["data"][txcode]["day"]
                if len({r[0] for r in erows}) != len(erows) or len({r[0] for r in trows}) != len(
                    trows
                ):
                    raise ValueError("duplicate dates")
                e = {r[0]: r for r in erows}
                t = {r[0]: r for r in trows}
                if target.isoformat() not in e or target.isoformat() not in t:
                    raise DataUnavailableError("备用指数缺少目标交易日。")
                if previous not in e or previous not in t:
                    raise DataUnavailableError("备用指数缺少前一交易日。")
                for day in (previous, target.isoformat()):
                    ev = np.array([float(x) for x in e[day][1:7]])
                    # Tencent index amount: 10,000 CNY; both volumes: 100 shares.
                    tv = np.array([float(x) for x in t[day][1:6]] + [float(t[day][8]) * 10000])
                    if not np.isfinite(ev).all() or not np.isfinite(tv).all():
                        raise ValueError("nonfinite")
                    if not np.allclose(ev, tv, rtol=0.0001, atol=0.011):
                        raise DataQualityError("东财/腾讯指数核验不一致，禁止入库。")
                row = e[target.isoformat()]
                values = [float(x) for x in row[1:7]]
                op, close, high, low, lots, amount = values
                if (
                    min(op, close, high, low) <= 0
                    or low > min(op, close)
                    or high < max(op, close)
                    or lots < 0
                    or amount < 0
                ):
                    raise ValueError("invalid OHLCV")
                if (lots == 0) != (amount == 0) or not np.isclose(
                    lots * 100, round(lots * 100), rtol=0, atol=1e-5
                ):
                    raise ValueError("invalid volume units")
                records.append(
                    {
                        "symbol": code,
                        "trade_date": pd.Timestamp(target),
                        "open": op,
                        "high": high,
                        "low": low,
                        "close": close,
                        "prev_close": float(e[previous][2]),
                        "volume_shares": round(lots * 100),
                        "amount_cny": amount,
                        "turnover_pct": float("nan"),
                        "source": "akshare:eastmoney:indices:tx_verified",
                        "retrieved_at": now.astimezone(UTC).isoformat(),
                    }
                )
            except (KeyError, IndexError, TypeError, ValueError):
                raise DataQualityError("免费指数备用字段、身份或单位不符合合同。") from None
        return DailyIncrementBatch(
            frame=pd.DataFrame(records, columns=["symbol", *CANONICAL_DAILY_COLUMNS]),
            target_date=target,
            requested_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
            received_symbols=BAOSTOCK_CORE_INDEX_SYMBOLS,
            fetched_at=now,
            trace_ids=(),
            provider="akshare",
            cutoff_timestamp=cutoff_timestamp
            if cutoff_timestamp is not None
            else int(now.timestamp()),
            unit_contract_version="em-tx-index-shares-cny-v1",
            unit_resolution_method_version="em_tx_all_six_crosscheck-v1",
            amount_multiplier_to_cny="eastmoney=1;tencent=10000",
        )
