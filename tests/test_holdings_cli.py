from __future__ import annotations

import json
from pathlib import Path

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.cli.holdings import main
from ashare_lab.services.holding_ledger import (
    HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY,
    get_active_holding_portfolio,
)


def _repository(tmp_path: Path) -> SQLiteRepository:
    repo = SQLiteRepository(
        tmp_path / "research.db",
        Path(__file__).resolve().parents[1] / "migrations",
    )
    repo.initialize()
    return repo


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "holdings.json"
    path.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "600919",
                        "name": "江苏银行",
                        "entry_date": "2026-08-28",
                        "cost_price": None,
                        "stock_sleeve_weight": 1.0,
                        "account_weight": None,
                        "company_action_clear": True,
                        "company_action_clear_through": "2026-08-28",
                        "company_action_evidence_source": "user_reviewed_announcements",
                        "company_action_evidence_id": "local-check-20260828-600919",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_replace_requires_explicit_confirmation(tmp_path: Path, capsys: object) -> None:
    result = main(
        ["replace", "--file", str(_file(tmp_path)), "--holding-weeks", "4"],
        _repository=_repository(tmp_path),
    )

    assert result == 2


def test_local_json_replace_list_and_clear(tmp_path: Path, capsys: object) -> None:
    repository = _repository(tmp_path)
    path = _file(tmp_path)

    assert (
        main(
            [
                "replace",
                "--file",
                str(path),
                "--holding-weeks",
                "4",
                "--effective-at",
                "2026-08-28T21:00:00+08:00",
                "--yes",
            ],
            _repository=repository,
        )
        == 0
    )
    capsys.readouterr()
    assert main(["list"], _repository=repository) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["holding_portfolio_version"] == 1
    assert listed["positions"][0]["cost_price"] is None
    assert listed[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] == []
    assert listed["network_used"] is False
    portfolio = get_active_holding_portfolio(repository)
    assert portfolio is not None
    assert portfolio.positions[0].metadata["company_action_clear"] is True
    assert portfolio.positions[0].metadata["company_action_clear_through"] == "2026-08-28"

    assert main(["clear", "--yes"], _repository=repository) == 0
    cleared = json.loads(capsys.readouterr().out)
    assert cleared["status"] == "cleared"
    assert cleared["positions"] == []


def test_replace_saves_only_explicit_per_provider_summary_consent(
    tmp_path: Path,
    capsys: object,
) -> None:
    repository = _repository(tmp_path)

    assert (
        main(
            [
                "replace",
                "--file",
                str(_file(tmp_path)),
                "--holding-weeks",
                "4",
                "--allow-holding-summary-bark",
                "--yes",
            ],
            _repository=repository,
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] == ["bark"]
    portfolio = get_active_holding_portfolio(repository)
    assert portfolio is not None
    assert portfolio.metadata[HOLDING_SUMMARY_DELIVERY_CHANNELS_KEY] == ["bark"]
    assert "external_delivery_consent" not in portfolio.metadata
