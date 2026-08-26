from __future__ import annotations

from types import SimpleNamespace

from ashare_lab.adapters.macos_keychain import (
    INFOWAY_KEYCHAIN_SERVICE,
    load_infoway_api_key,
    save_infoway_api_key,
)


def test_save_passes_secret_over_stdin_not_process_arguments(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    seen = {}

    def runner(args, **kwargs):
        seen["args"] = args
        seen["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    save_infoway_api_key("private-value", runner=runner)
    assert "private-value" not in " ".join(seen["args"])
    assert seen["input"] == "private-value\n"
    assert INFOWAY_KEYCHAIN_SERVICE in seen["args"]


def test_load_returns_none_when_keychain_item_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")

    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=44, stdout="", stderr="missing")

    assert load_infoway_api_key(runner=runner) is None
