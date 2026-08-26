from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_path

from ashare_lab.adapters.sqlite_repository import SQLiteRepository


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def application_data_dir() -> Path:
    path = Path(user_data_path("A股研究助手", appauthor=False, ensure_exists=True))
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_repository() -> SQLiteRepository:
    repository = SQLiteRepository(
        application_data_dir() / "research.db",
        project_root() / "migrations",
    )
    repository.initialize()
    return repository


def build_market_provider(name: str = "yahoo"):
    # Kept lazy so the UI can open and explain installation errors even when
    # the optional personal-research adapter is absent.
    normalized = name.strip().lower()
    if normalized == "akshare":
        from ashare_lab.adapters.akshare_market import AKShareMarketData

        return AKShareMarketData(application_data_dir() / "cache")
    if normalized == "yahoo":
        from ashare_lab.adapters.yfinance_market import YFinanceMarketData

        return YFinanceMarketData(application_data_dir() / "cache")
    raise ValueError(f"未知行情源：{name}")
