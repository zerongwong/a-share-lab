from __future__ import annotations

from ashare_lab.adapters.macos_keychain import (
    BARK_KEYCHAIN_SERVICE,
    INFOWAY_KEYCHAIN_SERVICE,
    SERVERCHAN_KEYCHAIN_SERVICE,
    delete_bark_device_key,
    load_infoway_api_key,
    save_bark_device_key,
    save_infoway_api_key,
    save_serverchan_sendkey,
)


def test_save_uses_native_backend_and_verifies_round_trip(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    stored = {}

    def setter(service, account, value):
        stored[(service, account)] = value

    def getter(service, account):
        return stored.get((service, account))

    save_infoway_api_key("private-value", setter=setter, getter=getter)
    assert list(stored.values()) == ["private-value"]
    assert next(iter(stored))[0] == INFOWAY_KEYCHAIN_SERVICE


def test_load_returns_none_when_keychain_item_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")

    def getter(_service, _account):
        return None

    assert load_infoway_api_key(getter=getter) is None


def test_notification_secrets_use_distinct_services(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    stored = {}

    def setter(service, account, value):
        stored[(service, account)] = value

    def getter(service, account):
        return stored.get((service, account))

    save_serverchan_sendkey("SCT-private-value", setter=setter, getter=getter)
    save_bark_device_key("bark-private-value", setter=setter, getter=getter)

    assert len(stored) == 2
    assert {service for service, _account in stored} == {
        SERVERCHAN_KEYCHAIN_SERVICE,
        BARK_KEYCHAIN_SERVICE,
    }


def test_save_rejects_false_positive_backend_result(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")

    def setter(_service, _account, _value):
        return None

    def getter(_service, _account):
        return ""

    import pytest

    with pytest.raises(Exception, match="保存后校验失败"):
        save_serverchan_sendkey(
            "SCT-private-value",
            setter=setter,
            getter=getter,
            deleter=lambda _service, _account: None,
        )


def test_save_removes_legacy_empty_item_before_write(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    state = {"value": ""}
    calls: list[str] = []

    def getter(_service, _account):
        return state["value"]

    def deleter(_service, _account):
        calls.append("delete")
        state["value"] = None

    def setter(_service, _account, value):
        calls.append("set")
        state["value"] = value

    save_serverchan_sendkey(
        "SCT-private-value",
        setter=setter,
        getter=getter,
        deleter=deleter,
    )

    assert calls == ["delete", "set"]
    assert state["value"] == "SCT-private-value"


def test_delete_missing_notification_key_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")

    deleted = []

    def getter(_service, _account):
        return None

    def deleter(service, _account):
        deleted.append(service)

    delete_bark_device_key(getter=getter, deleter=deleter)
    assert deleted == []
