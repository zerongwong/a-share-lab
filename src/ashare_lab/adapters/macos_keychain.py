"""Minimal macOS Keychain storage for provider credentials.

Secrets are written through the native keyring backend.  They are never placed
in process arguments, logs, project files, or environment variables.
"""

from __future__ import annotations

import getpass
import hmac
import sys
from collections.abc import Callable
from contextlib import suppress

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from ashare_lab.domain.errors import DataUnavailableError

INFOWAY_KEYCHAIN_SERVICE = "io.openai.asharelab.infoway-api"
TUSHARE_KEYCHAIN_SERVICE = "io.openai.asharelab.tushare-token"
SERVERCHAN_KEYCHAIN_SERVICE = "io.openai.asharelab.serverchan-sendkey"
BARK_KEYCHAIN_SERVICE = "io.openai.asharelab.bark-device-key"
CLOUDFLARE_R2_ACCESS_KEY_ID_KEYCHAIN_SERVICE = "io.openai.asharelab.cloudflare-r2-access-key-id"
CLOUDFLARE_R2_SECRET_ACCESS_KEY_KEYCHAIN_SERVICE = (
    "io.openai.asharelab.cloudflare-r2-secret-access-key"
)


SecretGetter = Callable[[str, str], str | None]
SecretSetter = Callable[[str, str, str], None]
SecretDeleter = Callable[[str, str], None]


def save_infoway_api_key(
    api_key: str,
    *,
    setter: SecretSetter = keyring.set_password,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _save_secret(
        INFOWAY_KEYCHAIN_SERVICE,
        api_key,
        empty_error="API Key不能为空",
        save_error="无法将Infoway密钥保存到macOS钥匙串。",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )


def load_infoway_api_key(
    *,
    getter: SecretGetter = keyring.get_password,
) -> str | None:
    return _load_secret(INFOWAY_KEYCHAIN_SERVICE, getter=getter)


def infoway_key_is_configured() -> bool:
    return load_infoway_api_key() is not None


def save_tushare_token(
    token: str,
    *,
    setter: SecretSetter = keyring.set_password,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    """Store the personal Tushare token without exposing it to project files."""

    _save_secret(
        TUSHARE_KEYCHAIN_SERVICE,
        token,
        empty_error="Tushare Token不能为空",
        save_error="无法将Tushare Token保存到macOS钥匙串。",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )


def load_tushare_token(
    *,
    getter: SecretGetter = keyring.get_password,
) -> str | None:
    return _load_secret(TUSHARE_KEYCHAIN_SERVICE, getter=getter)


def tushare_token_is_configured() -> bool:
    return load_tushare_token() is not None


def save_serverchan_sendkey(
    sendkey: str,
    *,
    setter: SecretSetter = keyring.set_password,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _save_secret(
        SERVERCHAN_KEYCHAIN_SERVICE,
        sendkey,
        empty_error="Server酱 SendKey不能为空",
        save_error="无法将Server酱 SendKey保存到macOS钥匙串。",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )


def load_serverchan_sendkey(
    *,
    getter: SecretGetter = keyring.get_password,
) -> str | None:
    return _load_secret(SERVERCHAN_KEYCHAIN_SERVICE, getter=getter)


def serverchan_key_is_configured() -> bool:
    return load_serverchan_sendkey() is not None


def delete_serverchan_sendkey(
    *,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _delete_secret(SERVERCHAN_KEYCHAIN_SERVICE, getter=getter, deleter=deleter)


def save_bark_device_key(
    device_key: str,
    *,
    setter: SecretSetter = keyring.set_password,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _save_secret(
        BARK_KEYCHAIN_SERVICE,
        device_key,
        empty_error="Bark设备Key不能为空",
        save_error="无法将Bark设备Key保存到macOS钥匙串。",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )


def load_bark_device_key(
    *,
    getter: SecretGetter = keyring.get_password,
) -> str | None:
    return _load_secret(BARK_KEYCHAIN_SERVICE, getter=getter)


def bark_key_is_configured() -> bool:
    return load_bark_device_key() is not None


def delete_bark_device_key(
    *,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _delete_secret(BARK_KEYCHAIN_SERVICE, getter=getter, deleter=deleter)


def save_cloudflare_r2_access_key_id(
    access_key_id: str,
    *,
    setter: SecretSetter = keyring.set_password,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _save_secret(
        CLOUDFLARE_R2_ACCESS_KEY_ID_KEYCHAIN_SERVICE,
        access_key_id,
        empty_error="Cloudflare R2 Access Key ID不能为空",
        save_error="无法将Cloudflare R2 Access Key ID保存到macOS钥匙串。",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )


def load_cloudflare_r2_access_key_id(
    *,
    getter: SecretGetter = keyring.get_password,
) -> str | None:
    return _load_secret(CLOUDFLARE_R2_ACCESS_KEY_ID_KEYCHAIN_SERVICE, getter=getter)


def cloudflare_r2_access_key_id_is_configured() -> bool:
    return load_cloudflare_r2_access_key_id() is not None


def delete_cloudflare_r2_access_key_id(
    *,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _delete_secret(
        CLOUDFLARE_R2_ACCESS_KEY_ID_KEYCHAIN_SERVICE,
        getter=getter,
        deleter=deleter,
    )


def save_cloudflare_r2_secret_access_key(
    secret_access_key: str,
    *,
    setter: SecretSetter = keyring.set_password,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _save_secret(
        CLOUDFLARE_R2_SECRET_ACCESS_KEY_KEYCHAIN_SERVICE,
        secret_access_key,
        empty_error="Cloudflare R2 Secret Access Key不能为空",
        save_error="无法将Cloudflare R2 Secret Access Key保存到macOS钥匙串。",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )


def load_cloudflare_r2_secret_access_key(
    *,
    getter: SecretGetter = keyring.get_password,
) -> str | None:
    return _load_secret(CLOUDFLARE_R2_SECRET_ACCESS_KEY_KEYCHAIN_SERVICE, getter=getter)


def cloudflare_r2_secret_access_key_is_configured() -> bool:
    return load_cloudflare_r2_secret_access_key() is not None


def cloudflare_r2_credentials_are_configured(
    *,
    getter: SecretGetter = keyring.get_password,
) -> bool:
    """Require both independent Keychain items before enabling an R2 client."""

    return bool(
        load_cloudflare_r2_access_key_id(getter=getter)
        and load_cloudflare_r2_secret_access_key(getter=getter)
    )


def delete_cloudflare_r2_secret_access_key(
    *,
    getter: SecretGetter = keyring.get_password,
    deleter: SecretDeleter = keyring.delete_password,
) -> None:
    _delete_secret(
        CLOUDFLARE_R2_SECRET_ACCESS_KEY_KEYCHAIN_SERVICE,
        getter=getter,
        deleter=deleter,
    )


def _save_secret(
    service: str,
    secret: str,
    *,
    empty_error: str,
    save_error: str,
    setter: SecretSetter,
    getter: SecretGetter,
    deleter: SecretDeleter,
) -> None:
    value = secret.strip()
    if not value:
        raise ValueError(empty_error)
    _require_macos()
    account = getpass.getuser()
    try:
        # Versions before 0.1 wrote an empty item through the command-line
        # ``security`` tool.  That item's access control can prevent the native
        # backend from updating it.  Remove only this known-empty legacy item;
        # a real stored credential is never deleted during save.
        if getter(service, account) == "":
            with suppress(PasswordDeleteError):
                deleter(service, account)
        setter(service, account, value)
        saved_value = getter(service, account)
    except KeyringError as exc:
        raise DataUnavailableError(save_error) from exc
    # A successful return from a credential backend is not enough.  Read back
    # immediately so the UI never reports a false-positive save.
    if not saved_value or not hmac.compare_digest(saved_value, value):
        raise DataUnavailableError(save_error + " 保存后校验失败，请重试。")


def _load_secret(
    service: str,
    *,
    getter: SecretGetter,
) -> str | None:
    _require_macos()
    try:
        raw_value = getter(service, getpass.getuser())
    except KeyringError as exc:
        raise DataUnavailableError("无法读取macOS钥匙串，请确认登录钥匙串已解锁。") from exc
    value = str(raw_value).strip() if raw_value is not None else ""
    return value or None


def _delete_secret(
    service: str,
    *,
    getter: SecretGetter,
    deleter: SecretDeleter,
) -> None:
    _require_macos()
    account = getpass.getuser()
    try:
        if getter(service, account) is None:
            return
        deleter(service, account)
    except PasswordDeleteError:
        # Idempotent deletion: the item may have disappeared between get/delete.
        return
    except KeyringError as exc:
        raise DataUnavailableError("无法从macOS钥匙串删除通知密钥。") from exc


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise DataUnavailableError("当前系统不是macOS，无法使用系统钥匙串。")
