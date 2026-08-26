"""Minimal macOS Keychain storage for provider credentials.

The secret is passed to ``security`` through standard input, never as a command
argument.  Callers must not log subprocess stdout from the read operation.
"""

from __future__ import annotations

import getpass
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from ashare_lab.domain.errors import DataUnavailableError

INFOWAY_KEYCHAIN_SERVICE = "io.openai.asharelab.infoway-api"


def _default_runner(args: list[str], **kwargs: Any):
    return subprocess.run(args, **kwargs)  # noqa: S603 - fixed macOS binary and fixed flags


def save_infoway_api_key(
    api_key: str,
    *,
    runner: Callable[..., Any] = _default_runner,
) -> None:
    value = api_key.strip()
    if not value:
        raise ValueError("API Key不能为空")
    _require_macos()
    result = runner(
        [
            "/usr/bin/security",
            "add-generic-password",
            "-a",
            getpass.getuser(),
            "-s",
            INFOWAY_KEYCHAIN_SERVICE,
            "-U",
            "-w",
        ],
        input=value + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise DataUnavailableError("无法将Infoway密钥保存到macOS钥匙串。")


def load_infoway_api_key(
    *,
    runner: Callable[..., Any] = _default_runner,
) -> str | None:
    _require_macos()
    result = runner(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-a",
            getpass.getuser(),
            "-s",
            INFOWAY_KEYCHAIN_SERVICE,
            "-w",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = str(result.stdout).strip()
    return value or None


def infoway_key_is_configured() -> bool:
    return load_infoway_api_key() is not None


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise DataUnavailableError("当前系统不是macOS，无法使用系统钥匙串。")
