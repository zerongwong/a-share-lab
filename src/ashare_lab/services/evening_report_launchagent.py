"""Render independent weekday report and deadline-watchdog tasks."""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

EVENING_REPORT_LAUNCHAGENT_LABEL = "com.zerong.asharelab.evening-report"
EVENING_REPORT_MODULE = "ashare_lab.cli.evening_report"
PREOPEN_DEADLINE_LAUNCHAGENT_LABEL = "com.zerong.asharelab.preopen-deadline"
PREOPEN_DEADLINE_MODULE = "ashare_lab.cli.preopen_deadline"
HOLDING_PNL_LAUNCHAGENT_LABEL = "com.zerong.asharelab.holding-pnl"
HOLDING_PNL_MODULE = "ashare_lab.cli.holding_pnl"
# macOS ``launchd`` follows cron weekday numbering: 1 is Monday and 5 is
# Friday.  The CLI still verifies that today is the next official trading
# session; weekday scheduling alone is never treated as market-calendar proof.
EVENING_REPORT_SCHEDULE = [
    {"Weekday": weekday, "Hour": hour, "Minute": minute}
    for weekday in range(1, 6)
    for hour, minute in ((8, 20), (8, 30), (8, 40), (8, 50))
]
PREOPEN_DEADLINE_SCHEDULE = [
    {"Weekday": weekday, "Hour": 8, "Minute": minute}
    for weekday in range(1, 6)
    for minute in (58, 59)
]
HOLDING_PNL_SCHEDULE = [
    {"Weekday": weekday, "Hour": 15, "Minute": minute}
    for weekday in range(1, 6)
    for minute in (35, 45, 55)
]


def render_evening_report_launchagent_plist(
    template_path: str | Path,
    output_path: str | Path,
    python_bin: str | Path,
    project_root: str | Path,
) -> Path:
    return _render_launchagent(
        template_path,
        output_path,
        python_bin,
        project_root,
        label=EVENING_REPORT_LAUNCHAGENT_LABEL,
        module=EVENING_REPORT_MODULE,
        schedule=EVENING_REPORT_SCHEDULE,
    )


def render_preopen_deadline_launchagent_plist(
    template_path: str | Path,
    output_path: str | Path,
    python_bin: str | Path,
    project_root: str | Path,
) -> Path:
    return _render_launchagent(
        template_path,
        output_path,
        python_bin,
        project_root,
        label=PREOPEN_DEADLINE_LAUNCHAGENT_LABEL,
        module=PREOPEN_DEADLINE_MODULE,
        schedule=PREOPEN_DEADLINE_SCHEDULE,
    )


def render_holding_pnl_launchagent_plist(
    template_path: str | Path,
    output_path: str | Path,
    python_bin: str | Path,
    project_root: str | Path,
) -> Path:
    return _render_launchagent(
        template_path,
        output_path,
        python_bin,
        project_root,
        label=HOLDING_PNL_LAUNCHAGENT_LABEL,
        module=HOLDING_PNL_MODULE,
        schedule=HOLDING_PNL_SCHEDULE,
    )


def _render_launchagent(
    template_path, output_path, python_bin, project_root, *, label, module, schedule
) -> Path:
    """Render an exact, secret-free LaunchAgent plist.

    The complete argument array is replaced through ``plistlib``.  This avoids
    the array-index insertion behaviour seen with ``plutil -replace`` on some
    macOS releases and preserves paths containing spaces.
    """

    source = Path(template_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    interpreter = os.path.abspath(os.fspath(Path(python_bin).expanduser()))
    working_directory = str(Path(project_root).expanduser().resolve())
    document = plistlib.loads(source.read_bytes())
    if not isinstance(document, dict) or document.get("Label") != label:
        raise ValueError("pre-open LaunchAgent template has an unexpected label")
    if "KeepAlive" in document:
        raise ValueError("evening report LaunchAgent must not contain KeepAlive")
    if document.get("StartCalendarInterval") != schedule:
        raise ValueError("pre-open LaunchAgent template has an unexpected weekday schedule")
    if document.get("RunAtLoad") is not True:
        raise ValueError("pre-open LaunchAgent must use RunAtLoad")
    # This deadline-sensitive computation reads the complete local market.
    # Background I/O and CPU throttling can stretch a normal two-minute build
    # past its watchdog. Keep the process ordinary while retaining its budget.
    document.pop("LowPriorityIO", None)
    document.pop("Nice", None)
    document["ProcessType"] = "Standard"
    document["ProgramArguments"] = [
        "/usr/bin/caffeinate",
        "-i",
        interpreter,
        "-m",
        module,
    ]
    document["WorkingDirectory"] = working_directory
    destination.write_bytes(plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=False))
    os.chmod(destination, 0o600)
    return destination
