"""Cross-process, re-entrant lock for holding-code disclosure and mutations."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock, local

_REGISTRY_GUARD = Lock()
_REGISTRY: dict[Path, RLock] = {}
_THREAD_LOCKS = local()


@contextmanager
def company_action_lock(path: Path, *, blocking: bool) -> Iterator[bool]:
    """Serialize refreshes with grant and canonical holding-ledger changes."""

    resolved = Path(path).expanduser().resolve()
    directory = resolved.parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    lock_path = directory / "refresh.lock"
    with _REGISTRY_GUARD:
        process_lock = _REGISTRY.setdefault(lock_path, RLock())
    process_acquired = process_lock.acquire(blocking=blocking)
    if not process_acquired:
        yield False
        return
    held = getattr(_THREAD_LOCKS, "paths", set())
    if lock_path in held:
        try:
            yield True
        finally:
            process_lock.release()
        return
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(lock_path, 0o600)
        operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
            acquired = True
            _THREAD_LOCKS.paths = {*held, lock_path}
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            _THREAD_LOCKS.paths = held
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        if descriptor is not None:
            os.close(descriptor)
        process_lock.release()
