"""Private child protocol for bounded SSE/SZSE listed A-share metadata."""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import pandas as pd

from ashare_lab.domain.errors import DataQualityError

_SH = re.compile(r"^6\d{5}$")
_SZ = re.compile(r"^(?:00|30)\d{4}$")


def _codes(frame, *, code_column, date_column, pattern, suffix, now):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataQualityError("official exchange stock list is empty")
    if code_column not in frame or date_column not in frame:
        raise DataQualityError("official exchange stock-list schema changed")
    codes = frame[code_column].astype("string").str.strip()
    if codes.isna().any() or not codes.str.fullmatch(pattern).all() or codes.duplicated().any():
        raise DataQualityError("official exchange stock identities are invalid")
    listed = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    if listed.isna().any() or (listed > now.date()).any():
        raise DataQualityError("official exchange listing dates are invalid")
    return tuple(code + suffix for code in codes)


def _read(now):
    import akshare as ak

    main = ak.stock_info_sh_name_code(symbol="主板A股")
    star = ak.stock_info_sh_name_code(symbol="科创板")
    sz = ak.stock_info_sz_name_code(symbol="A股列表")
    sh_main = _codes(
        main,
        code_column="证券代码",
        date_column="上市日期",
        pattern=_SH,
        suffix=".SH",
        now=now,
    )
    sh_star = _codes(
        star,
        code_column="证券代码",
        date_column="上市日期",
        pattern=_SH,
        suffix=".SH",
        now=now,
    )
    sz_a = _codes(
        sz,
        code_column="A股代码",
        date_column="A股上市日期",
        pattern=_SZ,
        suffix=".SZ",
        now=now,
    )
    if set(sh_main) & set(sh_star):
        raise DataQualityError("official Shanghai boards overlap")
    symbols = tuple(sorted((*sh_main, *sh_star, *sz_a)))
    if len(symbols) != len(set(symbols)) or not 4_000 <= len(symbols) < 6_000:
        raise DataQualityError("official exchange stock-list size is implausible")
    return {
        "symbols": symbols,
        "counts": {"sse_main": len(sh_main), "sse_star": len(sh_star), "szse_a": len(sz_a)},
        "retrieved_at": now.isoformat(),
        "provider": "sse_szse_official_via_akshare",
        "method_version": "exchange-listed-a-v1",
    }


def main():
    try:
        request = json.loads(sys.stdin.read())
        if request.get("operation") != "listed_a_share_symbols":
            raise ValueError("unsupported operation")
        now = datetime.fromisoformat(request["now"])
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("aware time required")
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            result = _read(now)
        print(json.dumps({"result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps({"error": "quality" if isinstance(exc, DataQualityError) else "unavailable"})
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
