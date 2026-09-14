"""Run the optional BaoStock SDK outside the process holding the update lock.

The SDK can block forever in socket.recv (including login/logout). A fresh
process per operation provides a wall-clock deadline without changing global
socket defaults, monkey-patching the SDK, or leaving timed-out threads alive.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime
from io import StringIO

import pandas as pd

from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.ports.daily_increment import DailyIncrementBatch


class BoundedBaoStockEod:
    provider = "baostock"

    def __init__(self, *, clock=None, timeout=45.0, runner=subprocess.run):
        if not 0 < timeout <= 120:
            raise ValueError("BaoStock deadline must be in (0, 120] seconds")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout = timeout
        self._runner = runner

    def _call(self, operation, **arguments):
        request = {"operation": operation, "now": self._clock().isoformat(), **arguments}
        try:
            result = self._runner(
                [sys.executable, "-m", "ashare_lab.cli.baostock_read"],
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DataUnavailableError("BaoStock请求超过时限，隔离进程已停止。") from None
        except OSError:
            raise DataUnavailableError("BaoStock隔离进程不可用。") from None
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise DataUnavailableError("BaoStock隔离响应不可用。") from None
        if not isinstance(payload, dict) or not ({"result", "error"} & payload.keys()):
            raise DataUnavailableError("BaoStock隔离响应结构无效。")
        if payload.get("error") == "quality":
            raise DataQualityError("BaoStock数据质量未通过，禁止切源掩盖。")
        if result.returncode or payload.get("error"):
            raise DataUnavailableError("BaoStock数据请求不可用。")
        return payload["result"]

    def fetch_cn_trading_days(self, start: date, end: date):
        return tuple(
            date.fromisoformat(d)
            for d in self._call("calendar", start=start.isoformat(), end=end.isoformat())
        )

    def fetch_cn_stock_symbols(self):
        return tuple(self._call("symbols"))

    def fetch_core_index_daily(self, target_date, *, cutoff_timestamp=None):
        data = self._call(
            "indices", target_date=target_date.isoformat(), cutoff_timestamp=cutoff_timestamp
        )
        data["frame"] = pd.read_json(StringIO(data["frame"]), orient="table")
        data["target_date"] = date.fromisoformat(data["target_date"])
        data["fetched_at"] = datetime.fromisoformat(data["fetched_at"])
        for field in ("requested_symbols", "received_symbols", "trace_ids", "metadata_sources"):
            data[field] = tuple(data[field])
        return DailyIncrementBatch(**data)
