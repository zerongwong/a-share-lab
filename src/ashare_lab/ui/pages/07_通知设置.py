from __future__ import annotations

import streamlit as st

from ashare_lab.adapters.macos_keychain import (
    bark_key_is_configured,
    load_bark_device_key,
    load_serverchan_sendkey,
    save_bark_device_key,
    save_serverchan_sendkey,
    serverchan_key_is_configured,
)
from ashare_lab.adapters.notification_channels import (
    BarkNotificationChannel,
    ServerChanNotificationChannel,
    normalize_bark_device_key,
    normalize_serverchan_sendkey,
)
from ashare_lab.domain.errors import AShareLabError
from ashare_lab.ports.notifications import NotificationMessage, NotificationUrgency

_FLASH_KEY = "notification_settings_flash"


def _test_message(channel: str) -> NotificationMessage:
    return NotificationMessage(
        title="A股研究室通知受理测试",
        body=(
            f"{channel}服务商已受理本次测试请求；这不等于终端已经送达。"
            "本消息不包含股票建议，也不会触发任何交易。"
        ),
        urgency=NotificationUrgency.NORMAL,
    )


def _show_flash() -> None:
    message = st.session_state.pop(_FLASH_KEY, None)
    if message:
        st.success(str(message))


def _configured_status(checker) -> bool | None:
    try:
        return bool(checker())
    except AShareLabError as exc:
        st.error(str(exc))
        return None


def _status_label(configured: bool | None) -> str:
    if configured is True:
        return "🟢 已配置"
    if configured is False:
        return "🟡 未配置"
    return "🔴 钥匙串暂时无法读取"


def _save_serverchan(value: str) -> None:
    normalized = normalize_serverchan_sendkey(value)
    save_serverchan_sendkey(normalized)
    st.session_state.pop("serverchan_sendkey_input", None)
    st.session_state[_FLASH_KEY] = "Server酱 SendKey已安全保存到macOS钥匙串。"
    st.rerun()


def _save_bark(value: str) -> None:
    normalized = normalize_bark_device_key(value)
    save_bark_device_key(normalized)
    st.session_state.pop("bark_device_key_input", None)
    st.session_state[_FLASH_KEY] = "Bark设备Key已安全保存到macOS钥匙串。"
    st.rerun()


def _test_serverchan() -> None:
    sendkey = load_serverchan_sendkey()
    if not sendkey:
        raise ValueError("请先保存Server酱 SendKey")
    with ServerChanNotificationChannel(sendkey) as channel:
        channel.send(_test_message("Server酱微信"))


def _test_bark() -> None:
    device_key = load_bark_device_key()
    if not device_key:
        raise ValueError("请先保存Bark设备Key")
    with BarkNotificationChannel(device_key) as channel:
        channel.send(_test_message("Bark iPhone"))


def _run_action(action, success_message: str) -> None:
    try:
        action()
    except (AShareLabError, ValueError) as exc:
        st.error(str(exc))
    else:
        st.success(success_message)


def render() -> None:
    st.title("通知设置")
    st.caption("密钥只保存在这台Mac的系统钥匙串，不写入项目、数据库、日志或GitHub。")
    _show_flash()

    st.info(
        "本页负责保存凭据和测试服务商是否受理。测试成功不代表微信或iPhone终端已经送达；"
        "周日至周四21:00晚报需要另行安装本机定时任务，系统不会自动下单。"
    )

    serverchan_configured = _configured_status(serverchan_key_is_configured)
    bark_configured = _configured_status(bark_key_is_configured)

    with st.container(border=True):
        st.subheader("Server酱 · 个人微信提醒")
        st.write("状态：" + _status_label(serverchan_configured))
        st.caption(
            "测试通过只表示Server酱服务商已受理，微信终端送达未确认。若微信没有收到，"
            "请到Server酱查看推送日志，并检查微信通道的关注或绑定状态。"
        )
        sendkey = st.text_input(
            "Server酱 Turbo SendKey或官方推送地址",
            type="password",
            key="serverchan_sendkey_input",
            placeholder="SCT…，或 https://sctapi.ftqq.com/SCT…",
            help="可粘贴纯SendKey或官方完整地址；请勿发送到聊天、截图或GitHub。",
        )
        save_column, test_column, docs_column = st.columns(3)
        if save_column.button("保存到钥匙串", key="save_serverchan", use_container_width=True):
            _run_action(lambda: _save_serverchan(sendkey), "Server酱配置已保存。")
        if test_column.button("发送微信测试", key="test_serverchan", use_container_width=True):
            _run_action(
                _test_serverchan,
                "Server酱服务商已受理测试消息；微信终端送达未确认。",
            )
        docs_column.link_button(
            "打开Server酱",
            "https://sct.ftqq.com/sendkey",
            use_container_width=True,
        )

    with st.container(border=True):
        st.subheader("Bark · iPhone完整行动单")
        st.write("状态：" + _status_label(bark_configured))
        st.caption(
            "在iPhone的Bark首页找到任意一张测试推送卡片，点“复制”；把复制出的完整"
            " https://api.day.app/… 地址直接粘贴到下面即可，无需自己寻找或截取Key。"
            "Bark与Server酱彼此独立，可作为微信通道的独立兜底。"
        )
        bark_value = st.text_input(
            "Bark设备Key或官方推送地址",
            type="password",
            key="bark_device_key_input",
            placeholder="设备Key，或 https://api.day.app/设备Key/…",
            help="当前仅接受Bark官方api.day.app地址；保存时只提取设备Key。",
        )
        save_column, test_column, docs_column = st.columns(3)
        if save_column.button("保存到钥匙串", key="save_bark", use_container_width=True):
            _run_action(lambda: _save_bark(bark_value), "Bark配置已保存。")
        if test_column.button("发送iPhone测试", key="test_bark", use_container_width=True):
            _run_action(
                _test_bark,
                "Bark服务商已受理测试消息；iPhone终端送达未确认。",
            )
        docs_column.link_button(
            "查看Bark说明",
            "https://github.com/Finb/Bark",
            use_container_width=True,
        )

    st.info(
        "晚报会分别尝试每个已配置通道；Server酱与Bark彼此独立，至少一个服务商受理后"
        "才登记去重状态。服务商受理不等于终端送达确认。"
    )
    st.warning("通知只提交衍生行动摘要。CSMAR原始数据、API凭据和完整持仓金额不会提交给第三方通道。")


render()
