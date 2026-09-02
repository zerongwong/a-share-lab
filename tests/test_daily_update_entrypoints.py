from __future__ import annotations

from contextlib import contextmanager
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
    assert "save_tushare_token" in source
    assert "macOS钥匙串" in source
    assert "不要发到聊天、截图或GitHub" in source
    assert "历史基线截止" in source
    assert "自动增量截止" in source
    assert "共同截止" in source
    assert "隔离失败" in source
    assert "不含北交所" in source
    assert "盘中不会把今天" in source
    assert "不完整数据会被隔离，18:30再复核" in source
    assert "三源字段、单位或交叉核验合同可能发生变化" in source
    assert "没有猜测单位，也没有用AKShare替换Tushare数据" in source
    assert "只有使用者主动运行安装脚本后" in source
    assert "每日15:30首次同步、18:30质量复核" in source
    assert "不依赖本网页" in source
    assert "daily_update_lock" in source
    assert "正在另一个进程中运行" in source
    assert "st.write(token)" not in source
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


def test_cli_returns_temporary_failure_when_another_update_owns_shared_lock(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    called = False

    @contextmanager
    def busy_lock():
        yield False

    def update(**_kwargs):
        nonlocal called
        called = True
        return _DummyReport()

    monkeypatch.setattr(sync_daily, "daily_update_lock", busy_lock)
    monkeypatch.setattr(sync_daily, "run_daily_update", update)

    status = sync_daily.main(
        [
            "--csmar-root",
            str(tmp_path / "csmar"),
            "--overlay-root",
            str(tmp_path / "overlay"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 75
    assert "另一个收盘数据更新正在运行" in captured.err
    assert called is False


def test_pyproject_registers_keychain_only_sync_command() -> None:
    source = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ashare-sync-daily = "ashare_lab.cli.sync_daily:main"' in source
    assert "daily_update_lock" in (PROJECT_ROOT / "src/ashare_lab/cli/sync_daily.py").read_text(
        encoding="utf-8"
    )
