from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from ashare_lab.cli import monthly_model_review as cli
from ashare_lab.services.build_monthly_model_review import MonthlyModelReview

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _empty_review() -> MonthlyModelReview:
    return MonthlyModelReview(
        review_month="2026-08",
        period_start=date(2026, 8, 1),
        period_end=date(2026, 8, 31),
        generated_as_of=date(2026, 9, 1),
        method_version="monthly-model-review-v0.1.0",
        horizon_reviews=(),
        experiment_proposals=(),
        excluded_batches=(),
        evidence_gaps=(),
        archive_scan_truncated=False,
        conclusion="样本不足，只列事实和证据缺口，不对模型优劣下结论，也不自动调整参数。",
    )


def test_cli_defaults_to_previous_completed_month_and_only_prints_json(
    monkeypatch,
    capsys,
) -> None:
    repository = object()
    observed = {}
    monkeypatch.setattr(cli, "build_repository", lambda: repository)

    def build(selected_repository, **kwargs):
        observed["repository"] = selected_repository
        observed.update(kwargs)
        return replace(_empty_review(), generated_as_of=kwargs["as_of"])

    monkeypatch.setattr(cli, "build_monthly_model_review", build)

    status = cli.main(["--as-of", "2026-09-01"])

    assert status == cli.EXIT_OK
    assert observed["repository"] is repository
    assert observed["review_month"] == date(2026, 8, 1)
    assert observed["as_of"] == date(2026, 9, 1)
    assert observed["report_limit"] == 1000
    payload = json.loads(capsys.readouterr().out)
    assert payload["review_month"] == "2026-08"
    assert payload["experiment_proposals"] == []


def test_cli_has_no_notification_secret_or_model_mutation_arguments() -> None:
    help_text = cli.build_parser().format_help().lower()

    assert "--send" not in help_text
    assert "--token" not in help_text
    assert "--api-key" not in help_text
    assert "--apply" not in help_text
    assert "--update-model" not in help_text
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ashare-monthly-model-review = "ashare_lab.cli.monthly_model_review:main"' in pyproject


def test_cli_fails_closed_without_leaking_details(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "build_repository",
        lambda: (_ for _ in ()).throw(RuntimeError("secret path and payload")),
    )

    status = cli.main(["--month", "2026-08", "--as-of", "2026-09-01"])

    captured = capsys.readouterr()
    assert status == cli.EXIT_ERROR
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert "secret" not in captured.err
