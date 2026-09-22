from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from ashare_lab.services.evening_report_launchagent import (
    EVENING_REPORT_LAUNCHAGENT_LABEL,
    EVENING_REPORT_MODULE,
    EVENING_REPORT_SCHEDULE,
    HOLDING_PNL_LAUNCHAGENT_LABEL,
    HOLDING_PNL_MODULE,
    HOLDING_PNL_SCHEDULE,
    PREOPEN_DEADLINE_LAUNCHAGENT_LABEL,
    PREOPEN_DEADLINE_MODULE,
    PREOPEN_DEADLINE_SCHEDULE,
    render_evening_report_launchagent_plist,
    render_holding_pnl_launchagent_plist,
    render_preopen_deadline_launchagent_plist,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLIST_TEMPLATE = PROJECT_ROOT / "config" / "com.zerong.asharelab.evening-report.plist.template"
DEADLINE_TEMPLATE = PROJECT_ROOT / "config" / "com.zerong.asharelab.preopen-deadline.plist.template"
PNL_TEMPLATE = PROJECT_ROOT / "config" / "com.zerong.asharelab.holding-pnl.plist.template"
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "install_evening_report_launchagent.sh"
UNINSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "uninstall_evening_report_launchagent.sh"


def test_report_template_is_independent_secret_free_and_weekday_preopen_only() -> None:
    document = plistlib.loads(PLIST_TEMPLATE.read_bytes())

    assert document["Label"] == EVENING_REPORT_LAUNCHAGENT_LABEL
    assert document["Label"] not in {
        "com.zerong.asharelab",
        "com.zerong.asharelab.daily-sync",
    }
    assert document["ProgramArguments"] == [
        "/usr/bin/caffeinate",
        "-i",
        "__PYTHON_BIN__",
        "-m",
        EVENING_REPORT_MODULE,
    ]
    assert document["RunAtLoad"] is True
    assert document["StartCalendarInterval"] == EVENING_REPORT_SCHEDULE
    assert [item["Weekday"] for item in document["StartCalendarInterval"]] == [
        day for day in range(1, 6) for _ in range(4)
    ]
    assert {(item["Hour"], item["Minute"]) for item in document["StartCalendarInterval"]} == {
        (8, 20),
        (8, 30),
        (8, 40),
        (8, 50),
    }
    assert document["ProcessType"] == "Standard"
    assert "LowPriorityIO" not in document
    assert "Nice" not in document
    assert 0 not in {item["Weekday"] for item in document["StartCalendarInterval"]}
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
        "/usr/bin/caffeinate",
        "-i",
        str(python_bin.absolute()),
        "-m",
        EVENING_REPORT_MODULE,
    ]
    assert len(document["ProgramArguments"]) == 5
    assert document["WorkingDirectory"] == str(project_root.resolve())
    assert document["StartCalendarInterval"] == EVENING_REPORT_SCHEDULE
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
    assert "render_preopen_deadline_launchagent_plist" in install
    assert PREOPEN_DEADLINE_MODULE in install
    assert PREOPEN_DEADLINE_LAUNCHAGENT_LABEL in install
    assert PREOPEN_DEADLINE_LAUNCHAGENT_LABEL in uninstall
    assert HOLDING_PNL_MODULE in install
    assert "render_holding_pnl_launchagent_plist" in install
    assert HOLDING_PNL_LAUNCHAGENT_LABEL in install
    assert HOLDING_PNL_LAUNCHAGENT_LABEL in uninstall
    assert "plutil -replace" not in install
    assert "previous.plist" in install
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
    assert "周一至周五08:20" in readme
    assert "当天是否交易日" in readme


def test_deadline_template_is_independent_and_runs_before_nine_only(tmp_path):
    output = tmp_path / "deadline.plist"
    render_preopen_deadline_launchagent_plist(DEADLINE_TEMPLATE, output, sys.executable, tmp_path)
    document = plistlib.loads(output.read_bytes())
    assert document["Label"] == PREOPEN_DEADLINE_LAUNCHAGENT_LABEL
    assert document["Label"] != EVENING_REPORT_LAUNCHAGENT_LABEL
    assert document["ProgramArguments"][-2:] == ["-m", PREOPEN_DEADLINE_MODULE]
    assert document["StartCalendarInterval"] == PREOPEN_DEADLINE_SCHEDULE
    assert len(PREOPEN_DEADLINE_SCHEDULE) == 10
    assert {(item["Hour"], item["Minute"]) for item in PREOPEN_DEADLINE_SCHEDULE} == {
        (8, 58),
        (8, 59),
    }
    assert document["RunAtLoad"] is True
    assert "KeepAlive" not in document
    assert document["EnvironmentVariables"]["ASHARE_EVENING_NOTIFICATION_CHANNELS"] == "serverchan"
    for value in ("token", "sendkey", "secret", "sct"):
        assert value not in DEADLINE_TEMPLATE.read_text().lower()
    assert output.stat().st_mode & 0o777 == 0o600


def test_pnl_template_has_own_label_and_three_postclose_attempts(tmp_path):
    output = tmp_path / "pnl.plist"
    render_holding_pnl_launchagent_plist(PNL_TEMPLATE, output, sys.executable, tmp_path)
    document = plistlib.loads(output.read_bytes())
    assert document["Label"] == HOLDING_PNL_LAUNCHAGENT_LABEL
    assert document["Label"] not in {
        EVENING_REPORT_LAUNCHAGENT_LABEL,
        PREOPEN_DEADLINE_LAUNCHAGENT_LABEL,
    }
    assert document["ProgramArguments"][-2:] == ["-m", HOLDING_PNL_MODULE]
    assert document["StartCalendarInterval"] == HOLDING_PNL_SCHEDULE
    assert len(HOLDING_PNL_SCHEDULE) == 15
    assert {(row["Hour"], row["Minute"]) for row in HOLDING_PNL_SCHEDULE} == {
        (15, 35),
        (15, 45),
        (15, 55),
    }
    assert document["RunAtLoad"] is True
    assert "KeepAlive" not in document
    assert output.stat().st_mode & 0o777 == 0o600


def _sandbox_installer(tmp_path, *, fail_watchdog, fail_pnl=False):
    """Replace launchctl with a local simulator; never touch actual launchd."""
    project = tmp_path / "project"
    scripts, config, target = project / "scripts", project / "config", tmp_path / "agents"
    for directory in (scripts, config, target, project / ".venv" / "bin"):
        directory.mkdir(parents=True)
    (project / ".venv" / "bin" / "python").symlink_to(sys.executable)
    for source in (PLIST_TEMPLATE, DEADLINE_TEMPLATE, PNL_TEMPLATE):
        (config / source.name).write_bytes(source.read_bytes())
    state_path = tmp_path / "launch-state.json"
    labels = (
        EVENING_REPORT_LAUNCHAGENT_LABEL,
        PREOPEN_DEADLINE_LAUNCHAGENT_LABEL,
        HOLDING_PNL_LAUNCHAGENT_LABEL,
    )
    state = {
        "loaded": [labels[0]],
        "disabled": [labels[1]],
        "fail_watchdog": fail_watchdog,
        "fail_pnl": fail_pnl,
    }
    state_path.write_text(json.dumps(state))
    originals = {}
    for label in labels:
        originals[label] = plistlib.dumps({"Label": label, "old_fixture": True})
        (target / f"{label}.plist").write_bytes(originals[label])
    command = tmp_path / "launchctl"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, plistlib, sys\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['ASHARE_TEST_LAUNCH_STATE'])\n"
        "state = json.loads(path.read_text())\n"
        "op, arg = sys.argv[1:3]\n"
        "label = arg.rsplit('/', 1)[-1]\n"
        "exit_code = 0\n"
        "if op == 'print': exit_code = 0 if label in state['loaded'] else 1\n"
        "elif op == 'print-disabled':\n"
        "    print('\\n'.join(chr(34)+x+chr(34)+' => true' for x in state['disabled']))\n"
        "elif op == 'bootout':\n"
        "    if label in state['loaded']: state['loaded'].remove(label)\n"
        "    else: exit_code = 1\n"
        "elif op in ('enable', 'disable'):\n"
        "    if label in state['disabled']: state['disabled'].remove(label)\n"
        "    if op == 'disable': state['disabled'].append(label)\n"
        "elif op == 'bootstrap':\n"
        "    document = plistlib.loads(Path(sys.argv[3]).read_bytes())\n"
        "    label = document['Label']\n"
        "    if (label.endswith('preopen-deadline') and state['fail_watchdog']) or (label.endswith('holding-pnl') and state['fail_pnl']):\n"
        "        state['fail_watchdog'] = False\n"
        "        state['fail_pnl'] = False\n"
        "        exit_code = 1\n"
        "    elif label not in state['loaded']: state['loaded'].append(label)\n"
        "path.write_text(json.dumps(state))\n"
        "sys.exit(exit_code)\n"
    )
    command.chmod(0o700)
    for source in (INSTALL_SCRIPT, UNINSTALL_SCRIPT):
        script = source.read_text().replace("/bin/launchctl", str(command))
        script = script.replace('TARGET_DIR="$HOME/Library/LaunchAgents"', f'TARGET_DIR="{target}"')
        # The other agent owns the CLI module. This test exercises transaction
        # rollback only; its import preflight is asserted separately above.
        script = script.replace(
            "import ashare_lab.cli.evening_report; import ashare_lab.cli.preopen_deadline; import ashare_lab.cli.holding_pnl",
            "pass",
        )
        script = script.replace('if [[ "$EUID" -eq 0 ]]; then', "if false; then")
        # macOS plutil is not present on Linux CI. plist content is asserted above.
        script = script.replace("/usr/bin/plutil -lint", "/usr/bin/true")
        (scripts / source.name).write_text(script)
    return scripts, target, state_path, originals


def test_installer_registers_all_and_uninstaller_removes_all_without_real_launchd(tmp_path):
    scripts, target, state_path, _ = _sandbox_installer(tmp_path, fail_watchdog=False)
    env = {
        **os.environ,
        "ASHARE_TEST_LAUNCH_STATE": str(state_path),
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    completed = subprocess.run(
        ["bash", str(scripts / INSTALL_SCRIPT.name)], env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads(state_path.read_text())
    assert set(state["loaded"]) == {
        EVENING_REPORT_LAUNCHAGENT_LABEL,
        PREOPEN_DEADLINE_LAUNCHAGENT_LABEL,
        HOLDING_PNL_LAUNCHAGENT_LABEL,
    }
    assert state["disabled"] == []
    subprocess.run(
        ["bash", str(scripts / UNINSTALL_SCRIPT.name)], env=env, check=True, capture_output=True
    )
    assert json.loads(state_path.read_text())["loaded"] == []
    assert list(target.glob("*.plist")) == []


@pytest.mark.parametrize("failed", ("deadline", "pnl"))
def test_install_failure_restores_all_files_and_prior_service_states(tmp_path, failed):
    scripts, target, state_path, originals = _sandbox_installer(
        tmp_path, fail_watchdog=failed == "deadline", fail_pnl=failed == "pnl"
    )
    completed = subprocess.run(
        ["bash", str(scripts / INSTALL_SCRIPT.name)],
        env={
            **os.environ,
            "ASHARE_TEST_LAUNCH_STATE": str(state_path),
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "恢复安装前状态" in completed.stderr
    for label, content in originals.items():
        assert (target / f"{label}.plist").read_bytes() == content
    state = json.loads(state_path.read_text())
    assert state["loaded"] == [EVENING_REPORT_LAUNCHAGENT_LABEL]
    assert state["disabled"] == [PREOPEN_DEADLINE_LAUNCHAGENT_LABEL]


def test_new_install_failure_removes_only_new_task_files(tmp_path):
    scripts, target, state_path, originals = _sandbox_installer(
        tmp_path, fail_watchdog=False, fail_pnl=True
    )
    for label in originals:
        (target / f"{label}.plist").unlink()
    state = json.loads(state_path.read_text())
    state.update({"loaded": [], "disabled": []})
    state_path.write_text(json.dumps(state))
    unrelated = target / "unrelated-service.plist"
    unrelated.write_text("unrelated fixture")
    completed = subprocess.run(
        ["bash", str(scripts / INSTALL_SCRIPT.name)],
        env={
            **os.environ,
            "ASHARE_TEST_LAUNCH_STATE": str(state_path),
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert list(target.glob("*.plist")) == [unrelated]
    assert unrelated.read_text() == "unrelated fixture"
    assert json.loads(state_path.read_text())["loaded"] == []
