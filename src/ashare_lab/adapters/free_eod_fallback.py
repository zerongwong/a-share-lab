"""Explicit free metadata/index failover. Never substitute on a quality failure.

Individual-stock source remains Tushare with independent AKShare verification:
historical web bars lacking exchange pre-close are not safe substitutes for it.
Index backup is Eastmoney unadjusted OHLCV/amount, checked against Tencent for
ALL six core indices. Index pre-close uses the immediately preceding verified
calendar session; this must never be reused as a stock ex-right reference.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from ashare_lab.adapters.baostock_eod import BAOSTOCK_CORE_INDEX_SYMBOLS
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import DailyIncrementBatch
from ashare_lab.ports.market_data import CANONICAL_DAILY_COLUMNS

_STOCK_MASTER_SYMBOL = re.compile(r"^(?:6\d{5}\.SH|(?:00|30)\d{4}\.SZ)$")
_MIN_CURRENT_STOCK_MASTER_SIZE = 5_000
_MAX_STOCK_MASTER_DRIFT_FRACTION = 0.005
_MAX_STOCK_MASTER_DRIFT_FLOOR = 50


def _validated_stock_master(values):
    try:
        symbols = tuple(values)
    except TypeError:
        raise DataQualityError("证券主表完整性证据不是可枚举的代码集合。") from None
    if (
        not symbols
        or any(
            not isinstance(symbol, str) or _STOCK_MASTER_SYMBOL.fullmatch(symbol) is None
            for symbol in symbols
        )
        or tuple(sorted(symbols)) != symbols
        or len(set(symbols)) != len(symbols)
    ):
        raise DataQualityError("证券主表完整性证据的身份、排序或唯一性无效。")
    return symbols


def _require_current_master_size(*masters):
    if any(len(master) < _MIN_CURRENT_STOCK_MASTER_SIZE for master in masters):
        raise DataQualityError("证券主表数量不足，禁止把可能截断的名单当作全市场。")


def _require_bounded_stock_master_drift(candidate, reference):
    candidate_symbols = _validated_stock_master(candidate)
    reference_symbols = _validated_stock_master(reference)
    _require_current_master_size(candidate_symbols, reference_symbols)
    drift = len(set(candidate_symbols).symmetric_difference(reference_symbols))
    allowed = max(
        _MAX_STOCK_MASTER_DRIFT_FLOOR,
        math.ceil(len(reference_symbols) * _MAX_STOCK_MASTER_DRIFT_FRACTION),
    )
    if drift > allowed:
        raise DataQualityError("证券主表相对最近已验证名单漂移过大。")
    return candidate_symbols, reference_symbols


def _require_exact_stock_master_agreement(candidate, reference):
    candidate_symbols = _validated_stock_master(candidate)
    reference_symbols = _validated_stock_master(reference)
    _require_current_master_size(candidate_symbols, reference_symbols)
    if candidate_symbols != reference_symbols:
        raise DataQualityError("BaoStock与交易所当前证券主表不一致，禁止发布。")
    return candidate_symbols


def _require_no_uncorroborated_contraction(candidate, reference):
    candidate_symbols, reference_symbols = _require_bounded_stock_master_drift(candidate, reference)
    if set(reference_symbols) - set(candidate_symbols):
        raise DataQualityError("证券主表相对最近已验证名单出现未经独立确认的缩减。")
    return candidate_symbols


class FreeEodMetadataFallback:
    def __init__(
        self,
        primary,
        metadata_backup,
        *,
        stock_master_backup=None,
        expected_stock_symbols=None,
        require_stock_master_completeness_evidence=False,
        clock=None,
        client=None,
    ):
        self.primary, self.backup = primary, metadata_backup
        self.stock_master_backup = (
            metadata_backup if stock_master_backup is None else stock_master_backup
        )
        self.expected_stock_symbols = (
            None
            if expected_stock_symbols is None
            else _validated_stock_master(expected_stock_symbols)
        )
        self.require_stock_master_completeness_evidence = bool(
            require_stock_master_completeness_evidence
        )
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
        try:
            result = self.primary.fetch_cn_stock_symbols()
        except DataUnavailableError:
            result, source = self._stock_master_after_primary_unavailable()
        else:
            result, source = self._validated_primary_stock_master(result)
        self.metadata_sources["fetch_cn_stock_symbols"] = source
        return result

    def _stock_master_after_primary_unavailable(self):
        if self.require_stock_master_completeness_evidence and self.expected_stock_symbols is None:
            raise DataUnavailableError(
                "首次证券主表同步缺少历史锚点，且BaoStock不可用；禁止用单一官方名单发布。"
            ) from None
        try:
            result = self.stock_master_backup.fetch_cn_stock_symbols()
        except DataUnavailableError:
            raise DataUnavailableError("证券主表所有免费来源均不可用。") from None
        if self.require_stock_master_completeness_evidence:
            # With BaoStock unavailable, the current official list still needs
            # the recent verified anchor.  Any contraction is ambiguous between
            # a real delisting and a truncated response, so fail closed until a
            # second current source can corroborate it.
            result = _require_no_uncorroborated_contraction(result, self.expected_stock_symbols)
        if self.stock_master_backup is self.backup:
            source = "tushare"
        else:
            source = getattr(self.stock_master_backup, "last_evidence", "") or getattr(
                self.stock_master_backup, "provider", "stock_master_backup"
            )
        return result, source

    def _validated_primary_stock_master(self, result):
        if not self.require_stock_master_completeness_evidence:
            return result, "baostock"
        candidate = _validated_stock_master(result)
        try:
            independent = self.stock_master_backup.fetch_cn_stock_symbols()
        except DataUnavailableError:
            if self.expected_stock_symbols is None:
                raise DataUnavailableError("首次证券主表同步缺少可用的独立完整性证据。") from None
            candidate = _require_no_uncorroborated_contraction(
                candidate, self.expected_stock_symbols
            )
            return candidate, "baostock:checked_against_last_verified_master"

        # A quality failure from the independent source intentionally escapes;
        # it must never be downgraded into an availability fallback.
        candidate = _require_exact_stock_master_agreement(candidate, independent)
        if self.expected_stock_symbols is not None:
            _require_bounded_stock_master_drift(independent, self.expected_stock_symbols)
        evidence = getattr(self.stock_master_backup, "last_evidence", "") or getattr(
            self.stock_master_backup, "provider", "stock_master_backup"
        )
        return candidate, f"baostock:crosschecked:{evidence}"

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
