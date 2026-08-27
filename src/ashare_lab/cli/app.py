"""User-friendly local launcher for the Streamlit research app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as streamlit_cli

    app = Path(__file__).resolve().parents[1] / "ui" / "A股研究室.py"
    if not app.is_file():
        raise RuntimeError("A股研究室界面文件未随安装包正确安装")
    sys.argv = [
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
