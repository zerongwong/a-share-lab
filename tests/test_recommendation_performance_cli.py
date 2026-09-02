from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ashare_lab.cli import recommendation_performance as cli
from ashare_lab.services.build_evening_digest import EveningResearchDigest
from ashare_lab.services.run_recommendation_performance import (
    RecommendationPerformanceRunSummary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFF = date(2026, 8, 27)
PLAN_DATE = date(2026, 8, 28)


def _summary(*, failures: tuple[str, ...] = (), notification_failures: int = 0):
    return RecommendationPerformanceRunSummary(
        latest_verified_date=PLAN_DATE,
        pending_batches=6,
        evaluated_batches=1,
        persisted_batches=1,
        mature_batches=1,
        notification_attempts=1,
        notification_accepted_batches=1 if not notification_failures else 0,
        notification_failed_batches=notification_failures,
        accepted_channels=("serverchan",) if not notification_failures else (),
        failed_batch_ids=failures,
    )


class _ManifestStore:
    def __init__(self, root: Path, *, predecessor: date = CUTOFF) -> None:
        self.root = root
        self.predecessor = predecessor

    def read_verified_manifest(self, *, source_id: str) -> pd.DataFrame:
        assert source_id == "zero_budget_eod"
        return pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp(PLAN_DATE),
                    "previous_trade_date": pd.Timestamp(self.predecessor),
                    "adjustment": "none",
                }
            ]
        )


def _digest() -> EveningResearchDigest:
    return EveningResearchDigest(
        common_cutoff=CUTOFF,
        decision_date=CUTOFF,
        cycle_label="历史重建",
        entry_strictness="defensive",
        max_stock_exposure=0.3,
        minimum_cash_weight=0.7,
        cycle_rule_agreement=None,
        periods=(),
    )


def test_default_command_settles_local_archive_and_uses_scheduled_notifier(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    repository = object()
    observed = {}
    monkeypatch.setattr(cli, "application_data_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "build_repository", lambda: repository)
    monkeypatch.setattr(cli, "MarketOverlayStore", lambda root: _ManifestStore(root))

    def run(**kwargs):
        observed.update(kwargs)
        return _summary()

    monkeypatch.setattr(cli, "run_recommendation_performance", run)

    status = cli.main([])

    assert status == cli.EXIT_OK
    assert observed["repository"] is repository
    assert observed["overlay_store"].root == (tmp_path / "cache" / "market_overlay").resolve()
    assert observed["notifier"] is cli.send_scheduled_notification
    assert observed["as_of"] is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["mature_batches"] == 1
    assert payload["accepted_channels"] == ["serverchan"]


def test_settle_returns_incomplete_when_any_batch_or_notification_failed(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "application_data_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "build_repository", object)
    monkeypatch.setattr(cli, "MarketOverlayStore", lambda root: _ManifestStore(root))
    monkeypatch.setattr(
        cli,
        "run_recommendation_performance",
        lambda **_kwargs: _summary(failures=("batch-1",), notification_failures=1),
    )

    status = cli.main(
        ["settle", "--overlay-root", str(tmp_path / "verified"), "--as-of", "2026-08-31"]
    )

    assert status == cli.EXIT_INCOMPLETE
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed_batch_ids"] == ["batch-1"]


def test_reconstruct_is_offline_explicit_and_can_only_archive_reconstructed(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    observed = {}
    repository = object()
    monkeypatch.setattr(cli, "application_data_dir", lambda: tmp_path / "app-data")
    monkeypatch.setattr(cli, "build_repository", lambda: repository)
    monkeypatch.setattr(cli, "MarketOverlayStore", lambda root: _ManifestStore(root))

    def build(**kwargs):
        observed["build"] = kwargs
        return _digest()

    def archive(digest, selected_repository, *, archive_nature):
        observed["archive"] = (digest, selected_repository, archive_nature)
        return SimpleNamespace(
            report_id="reconstructed-report",
            content_hash="reconstructed-hash",
            batches=(1, 2, 3, 4, 5, 6),
            members=(1, 2, 3, 4),
        )

    monkeypatch.setattr(cli, "build_evening_research_digest", build)
    monkeypatch.setattr(cli, "archive_recommendation_report", archive)
    monkeypatch.setattr(
        cli,
        "run_recommendation_performance",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not settle")),
    )
    monkeypatch.setattr(
        cli,
        "send_scheduled_notification",
        lambda _message: (_ for _ in ()).throw(AssertionError("must not notify")),
    )

    status = cli.main(
        [
            "reconstruct",
            "--decision-date",
            CUTOFF.isoformat(),
            "--plan-for-date",
            PLAN_DATE.isoformat(),
            "--csmar-root",
            str(tmp_path / "csmar"),
            "--overlay-root",
            str(tmp_path / "overlay"),
            "--reference-root",
            str(tmp_path / "reference"),
        ]
    )

    assert status == cli.EXIT_OK
    assert observed["build"]["decision_date"] == CUTOFF
    assert observed["build"]["dataset_root"] == (tmp_path / "csmar").resolve()
    archived_digest, archived_repository, archive_nature = observed["archive"]
    assert archived_digest.plan_for_date == PLAN_DATE
    assert archived_repository is repository
    assert archive_nature == "reconstructed"
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "archive_nature": "reconstructed",
        "batch_count": 6,
        "common_cutoff": "2026-08-27",
        "content_hash": "reconstructed-hash",
        "decision_date": "2026-08-27",
        "member_count": 4,
        "network_used": False,
        "notification_sent": False,
        "orders_enabled": False,
        "plan_for_date": "2026-08-28",
        "report_id": "reconstructed-report",
        "status": "archived_reconstructed",
    }


def test_reconstruct_fails_closed_when_local_session_chain_is_not_verified(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    archived = False
    monkeypatch.setattr(cli, "application_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "MarketOverlayStore",
        lambda root: _ManifestStore(root, predecessor=date(2026, 8, 26)),
    )
    monkeypatch.setattr(cli, "build_evening_research_digest", lambda **_kwargs: _digest())

    def archive(*_args, **_kwargs):
        nonlocal archived
        archived = True

    monkeypatch.setattr(cli, "archive_recommendation_report", archive)

    status = cli.main(
        [
            "reconstruct",
            "--decision-date",
            CUTOFF.isoformat(),
            "--plan-for-date",
            PLAN_DATE.isoformat(),
        ]
    )

    captured = capsys.readouterr()
    assert status == cli.EXIT_ERROR
    assert archived is False
    assert captured.out == ""
    assert "ValueError" in captured.err
    assert "2026" not in captured.err


def test_cli_has_no_secret_or_original_backfill_arguments_and_is_registered() -> None:
    help_text = cli.build_parser().format_help().lower()
    reconstruct_help = (
        cli.build_parser()
        ._subparsers._group_actions[0]
        .choices["reconstruct"]
        .format_help()
        .lower()
    )
    combined = help_text + reconstruct_help

    assert "--api-key" not in combined
    assert "--token" not in combined
    assert "--sendkey" not in combined
    assert "--archive-nature" not in combined
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "load_infoway_api_key" not in source
    assert 'archive_nature="reconstructed"' in source
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'ashare-recommendation-performance = "ashare_lab.cli.recommendation_performance:main"'
    ) in pyproject
