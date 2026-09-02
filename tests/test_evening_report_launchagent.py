from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

from ashare_lab.services.evening_report_launchagent import (
    EVENING_REPORT_LAUNCHAGENT_LABEL,
    EVENING_REPORT_MODULE,
    EVENING_REPORT_SCHEDULE,
    render_evening_report_launchagent_plist,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLIST_TEMPLATE = PROJECT_ROOT / "config" / "com.zerong.asharelab.evening-report.plist.template"
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "install_evening_report_launchagent.sh"
UNINSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "uninstall_evening_report_launchagent.sh"


def test_evening_report_template_is_independent_secret_free_and_sun_thu_21_only() -> None:
    document = plistlib.loads(PLIST_TEMPLATE.read_bytes())

    assert document["Label"] == EVENING_REPORT_LAUNCHAGENT_LABEL
    assert document["Label"] not in {
        "com.zerong.asharelab",
        "com.zerong.asharelab.daily-sync",
    }
    assert document["ProgramArguments"] == [
        "__PYTHON_BIN__",
        "-m",
        EVENING_REPORT_MODULE,
    ]
    assert document["RunAtLoad"] is True
    assert document["StartCalendarInterval"] == EVENING_REPORT_SCHEDULE
    assert [item["Weekday"] for item in document["StartCalendarInterval"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert all(item["Hour"] == 21 for item in document["StartCalendarInterval"])
    assert all(item["Minute"] == 0 for item in document["StartCalendarInterval"])
    assert 5 not in {item["Weekday"] for item in document["StartCalendarInterval"]}
    assert 6 not in {item["Weekday"] for item in document["StartCalendarInterval"]}
    assert "KeepAlive" not in document
    assert document["StandardOutPath"] == "/dev/null"
    assert document["StandardErrorPath"] == "/dev/null"
    assert document["EnvironmentVariables"]["ASHARE_EVENING_NOTIFICATION_CHANNELS"] == (
        "serverchan"
    )
    serialized = PLIST_TEMPLATE.read_text(encoding="utf-8").lower()
    for secret_name in ("api_key", "sendkey", "device_key", "token", "secret", "sct"):
        assert secret_name not in serialized


def test_evening_report_renderer_replaces_the_complete_argument_array(tmp_path: Path) -> None:
    output = tmp_path / "rendered.plist"
    project_root = tmp_path / "Project With Spaces"
    python_bin = project_root / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to(sys.executable)

    render_evening_report_launchagent_plist(
        PLIST_TEMPLATE,
        output,
        python_bin,
        project_root,
    )

    document = plistlib.loads(output.read_bytes())
    assert document["ProgramArguments"] == [
        str(python_bin.absolute()),
        "-m",
        EVENING_REPORT_MODULE,
    ]
    assert len(document["ProgramArguments"]) == 3
    assert document["WorkingDirectory"] == str(project_root.resolve())
    assert document["StartCalendarInterval"] == [
        {"Weekday": weekday, "Hour": 21, "Minute": 0} for weekday in range(0, 5)
    ]
    assert "__PYTHON_BIN__" not in document["ProgramArguments"]
    assert output.stat().st_mode & 0o777 == 0o600


def test_evening_report_scripts_are_syntax_valid_independent_and_rollback_safe() -> None:
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(UNINSTALL_SCRIPT)], check=True)
    install = INSTALL_SCRIPT.read_text(encoding="utf-8")
    uninstall = UNINSTALL_SCRIPT.read_text(encoding="utf-8")

    assert 'LABEL="com.zerong.asharelab.evening-report"' in install
    assert 'LABEL="com.zerong.asharelab.evening-report"' in uninstall
    assert "ashare_lab.cli.evening_report" in install
    assert "render_evening_report_launchagent_plist" in install
    assert "plutil -replace" not in install
    assert "BACKUP_PLIST" in install
    assert "restore_previous" in install
    assert "未能完成登记，正在恢复安装前状态" in install
    assert "launchctl bootstrap" in install
    assert "launchctl bootout" in uninstall
    assert "com.zerong.asharelab.daily-sync" not in install
    assert "com.zerong.asharelab.daily-sync" not in uninstall
    assert "com.zerong.asharelab.plist" not in install
    assert "com.zerong.asharelab.plist" not in uninstall
    assert "security delete" not in uninstall
    assert "research.db" not in uninstall


def test_readme_documents_manual_run_and_explicit_installation() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert ".venv/bin/python -m ashare_lab.cli.evening_report" in readme
    assert "./scripts/install_evening_report_launchagent.sh" in readme
    assert "./scripts/uninstall_evening_report_launchagent.sh" in readme
    assert "周日至周四21:00" in readme
    assert "周五、周六" in readme
