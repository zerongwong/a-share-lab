"""One shared advisory lock for every user-facing daily update entrypoint."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ashare_lab.bootstrap import application_data_dir


def default_daily_update_lock_path() -> Path:
    return application_data_dir() / "scheduler" / "daily-sync.lock"


@contextmanager
def daily_update_lock(path: str | Path | None = None) -> Iterator[bool]:
    """Acquire the common non-blocking lock used by UI and both CLIs.

    The kernel releases this advisory lock when the process exits, including
    abnormal termination.  The caller decides whether an already-running
    update is a harmless scheduled no-op or a temporary manual-command error.
    """

    lock_path = Path(path or default_daily_update_lock_path()).expanduser().resolve()
    _ensure_private_directory(lock_path.parent)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.chmod(lock_path, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
