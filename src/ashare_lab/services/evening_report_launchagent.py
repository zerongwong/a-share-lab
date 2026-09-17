"""Render the independent Monday-to-Friday 09:00 pre-open report task."""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

EVENING_REPORT_LAUNCHAGENT_LABEL = "com.zerong.asharelab.evening-report"
EVENING_REPORT_MODULE = "ashare_lab.cli.evening_report"
# macOS ``launchd`` follows cron weekday numbering: 1 is Monday and 5 is
# Friday.  The CLI still verifies that today is the next official trading
# session; weekday scheduling alone is never treated as market-calendar proof.
EVENING_REPORT_SCHEDULE = [
    {"Weekday": weekday, "Hour": 9, "Minute": minute}
    for weekday in range(1, 6)
    for minute in (0, 10, 20)
]


def render_evening_report_launchagent_plist(
    template_path: str | Path,
    output_path: str | Path,
    python_bin: str | Path,
    project_root: str | Path,
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
    if not isinstance(document, dict) or document.get("Label") != EVENING_REPORT_LAUNCHAGENT_LABEL:
        raise ValueError("evening report LaunchAgent template has an unexpected label")
    if "KeepAlive" in document:
        raise ValueError("evening report LaunchAgent must not contain KeepAlive")
    if document.get("StartCalendarInterval") != EVENING_REPORT_SCHEDULE:
        raise ValueError("pre-open report must retry at 09:00, 09:10 and 09:20 Monday-to-Friday")
    document["ProgramArguments"] = [
        "/usr/bin/caffeinate",
        "-i",
        interpreter,
        "-m",
        EVENING_REPORT_MODULE,
    ]
    document["WorkingDirectory"] = working_directory
    destination.write_bytes(plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=False))
    os.chmod(destination, 0o600)
    return destination
