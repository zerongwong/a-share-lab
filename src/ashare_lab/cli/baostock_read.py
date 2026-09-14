"""Private child protocol for bounded public BaoStock reads; no credentials."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import fields
from datetime import date, datetime

from ashare_lab.adapters.baostock_eod import BaoStockEodMarketData
from ashare_lab.domain.errors import DataQualityError


def main():
    try:
        request = json.loads(sys.stdin.read())
        provider = BaoStockEodMarketData(clock=lambda: datetime.fromisoformat(request["now"]))
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            if request["operation"] == "calendar":
                result = [
                    d.isoformat()
                    for d in provider.fetch_cn_trading_days(
                        date.fromisoformat(request["start"]), date.fromisoformat(request["end"])
                    )
                ]
            elif request["operation"] == "symbols":
                result = provider.fetch_cn_stock_symbols()
            elif request["operation"] == "indices":
                batch = provider.fetch_core_index_daily(
                    date.fromisoformat(request["target_date"]),
                    cutoff_timestamp=request.get("cutoff_timestamp"),
                )
                result = {f.name: getattr(batch, f.name) for f in fields(batch)}
                result["frame"] = batch.frame.to_json(orient="table", date_format="iso")
                result["target_date"] = batch.target_date.isoformat()
                result["fetched_at"] = batch.fetched_at.isoformat()
            else:
                raise ValueError("unsupported operation")
        print(json.dumps({"result": result}))
        return 0
    except Exception as exc:
        print(
            json.dumps({"error": "quality" if isinstance(exc, DataQualityError) else "unavailable"})
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
