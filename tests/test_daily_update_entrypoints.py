from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ashare_lab.cli import sync_daily
from ashare_lab.domain.errors import DataUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "src" / "ashare_lab" / "ui"
PAGE = UI_ROOT / "pages" / "09_自动数据更新.py"


def test_router_registers_automatic_daily_update_page() -> None:
    source = (UI_ROOT / "A股研究室.py").read_text(encoding="utf-8")

    assert '_existing_page("09")' in source
    assert 'title="自动数据更新"' in source
    assert 'url_path="automatic-daily-update"' in source


def test_update_page_masks_key_and_explains_all_fail_closed_boundaries() -> None:
    source = PAGE.read_text(encoding="utf-8")

    assert 'type="password"' in source
    assert "save_infoway_api_key" in source
    assert "macOS钥匙串" in source
    assert "旧密钥应先在Infoway后台轮换" in source
    assert "历史基线截止" in source
    assert "自动增量截止" in source
    assert "共同截止" in source
    assert "隔离失败" in source
    assert "不含北交所" in source
    assert "盘中不会把今天" in source
    assert "供应商字段契约变化，请更新适配器" in source
    assert "没有自动尝试vm、替换vw" in source
    assert "不会安装LaunchAgent" in source
    assert "st.write(api_key)" not in source
    assert "os.environ" not in source


def test_cli_has_no_secret_or_unit_guessing_arguments() -> None:
    help_text = sync_daily.build_parser().format_help().lower()

    assert "--csmar-root" in help_text
    assert "--overlay-root" in help_text
    assert "--api-key" not in help_text
    assert "--token" not in help_text
    assert "--amount-field" not in help_text
    assert "--volume-multiplier" not in help_text


@dataclass(frozen=True)
class _DummyReport:
    current_through_latest_complete_session: bool = True
    common_cutoff: str = "2026-08-26"


def test_cli_success_output_contains_report_but_no_key(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(sync_daily, "run_daily_update", lambda **_kwargs: _DummyReport())

    status = sync_daily.main(
        [
            "--csmar-root",
            str(tmp_path / "csmar"),
            "--overlay-root",
            str(tmp_path / "overlay"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "2026-08-26" in captured.out
    assert "api_key" not in captured.out.lower()
    assert captured.err == ""


def test_cli_failure_is_nonzero_without_secret_echo(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        sync_daily,
        "run_daily_update",
        lambda **_kwargs: (_ for _ in ()).throw(DataUnavailableError("钥匙串未配置")),
    )

    status = sync_daily.main(
        [
            "--csmar-root",
            str(tmp_path / "csmar"),
            "--overlay-root",
            str(tmp_path / "overlay"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "钥匙串未配置" in captured.err
    assert "api_key" not in captured.err.lower()


def test_pyproject_registers_keychain_only_sync_command() -> None:
    source = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ashare-sync-daily = "ashare_lab.cli.sync_daily:main"' in source
