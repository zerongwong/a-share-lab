"""Private, read-only Tushare calendar child; no notifications or state writes."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from datetime import date

from ashare_lab.domain.errors import DataQualityError


def main() -> int:
    try:
        request = json.loads(sys.stdin.read(4096))
        start = date.fromisoformat(request["start"])
        end = date.fromisoformat(request["end"])
        if start > end or (end - start).days >= 35:
            raise DataQualityError("invalid bounded calendar range")
        from ashare_lab.cli.evening_digest import _tushare_calendar_backup

        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            values = _tushare_calendar_backup(start, end)
        print(json.dumps({"sessions": [day.isoformat() for day in values]}))
        return 0
    except Exception as exc:
        print(
            json.dumps({"error": "quality" if isinstance(exc, DataQualityError) else "unavailable"})
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
