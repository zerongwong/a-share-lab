from __future__ import annotations

import streamlit as st

from ashare_lab.adapters.macos_keychain import (
    bark_key_is_configured,
    cloudflare_r2_access_key_id_is_configured,
    cloudflare_r2_secret_access_key_is_configured,
    load_bark_device_key,
    load_serverchan_sendkey,
    save_bark_device_key,
    save_cloudflare_r2_access_key_id,
    save_cloudflare_r2_secret_access_key,
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
from ashare_lab.services.chart_publisher_settings import (
    CLOUDFLARE_R2_PUBLISHER_ID,
    DEFAULT_R2_OBJECT_PREFIX,
    DEFAULT_SIGNED_URL_TTL_SECONDS,
    ChartPublisherSettings,
    load_chart_publisher_settings,
    save_chart_publisher_settings,
)

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


def _save_r2_coordinates(
    account_id: str,
    bucket_name: str,
    signed_url_ttl_seconds: int,
    private_bucket_verified: bool,
    lifecycle_rule_verified: bool,
) -> None:
    settings = ChartPublisherSettings(
        publisher_id=CLOUDFLARE_R2_PUBLISHER_ID,
        account_id=account_id,
        bucket_name=bucket_name,
        object_prefix=DEFAULT_R2_OBJECT_PREFIX,
        signed_url_ttl_seconds=signed_url_ttl_seconds,
        private_bucket_verified=private_bucket_verified,
        lifecycle_delete_after_days=1,
        lifecycle_rule_verified=lifecycle_rule_verified,
    )
    save_chart_publisher_settings(settings)
    st.session_state[_FLASH_KEY] = (
        "Cloudflare R2非敏感配置已保存在本机；只有私有桶和1日删除均确认后才允许发布。"
    )
    st.rerun()


def _save_r2_credentials(access_key_id: str, secret_access_key: str) -> None:
    if not access_key_id.strip() or not secret_access_key.strip():
        raise ValueError("请同时填写R2 Access Key ID和Secret Access Key")
    save_cloudflare_r2_access_key_id(access_key_id)
    save_cloudflare_r2_secret_access_key(secret_access_key)
    st.session_state.pop("r2_access_key_id_input", None)
    st.session_state.pop("r2_secret_access_key_input", None)
    st.session_state[_FLASH_KEY] = "Cloudflare R2两项密钥已安全保存到macOS钥匙串。"
    st.rerun()


def _load_r2_coordinates() -> ChartPublisherSettings | None:
    try:
        return load_chart_publisher_settings()
    except (OSError, TypeError, ValueError) as exc:
        st.error(f"R2本机配置暂时无法读取（{type(exc).__name__}）。")
        return None


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
    r2_access_key_configured = _configured_status(cloudflare_r2_access_key_id_is_configured)
    r2_secret_key_configured = _configured_status(cloudflare_r2_secret_access_key_is_configured)
    r2_settings = _load_r2_coordinates()

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

    with st.container(border=True):
        st.subheader("Cloudflare R2 · 持仓K线图私有发布配置")
        st.info(
            "当前生产晚报按你的选择只发送精简文字和持仓预警，图片外发已停用；"
            "因此无需开通或配置R2。下面的配置仅为历史兼容保留，不会被定时晚报调用。"
        )
        st.warning(
            "保存本页不会联网、上传或发送图片。"
            "存储桶必须保持私有，并在R2端配置1日自动删除；"
            "两项都未确认时，晚报会自动退回纯文字。"
        )
        st.caption(
            "Account ID、存储桶、对象目录和有效期不是密钥，只写入本机私有"
            "配置文件；Access Key ID和Secret Access Key只写入macOS钥匙串。"
        )
        st.write("非敏感配置：" + ("🟢 已保存" if r2_settings is not None else "🟡 未保存"))
        st.write("Access Key ID：" + _status_label(r2_access_key_configured))
        st.write("Secret Access Key：" + _status_label(r2_secret_key_configured))

        account_id = st.text_input(
            "Cloudflare Account ID",
            value="" if r2_settings is None else r2_settings.account_id,
            key="r2_account_id_input",
            help="32位Account ID；它不是API密钥。",
        )
        bucket_name = st.text_input(
            "R2存储桶名称",
            value="" if r2_settings is None else r2_settings.bucket_name,
            key="r2_bucket_name_input",
        )
        st.text_input(
            "R2对象目录（固定）",
            value=DEFAULT_R2_OBJECT_PREFIX,
            disabled=True,
            help="对象目录固定为holding-charts，不接受自定义。",
        )
        signed_url_ttl_seconds = int(
            st.number_input(
                "短期签名地址有效期（秒）",
                min_value=300,
                max_value=3_600,
                value=(
                    DEFAULT_SIGNED_URL_TTL_SECONDS
                    if r2_settings is None
                    else r2_settings.signed_url_ttl_seconds
                ),
                step=300,
                key="r2_signed_url_ttl_input",
            )
        )
        private_bucket_verified = st.checkbox(
            "我已确认该R2存储桶保持私有",
            value=(False if r2_settings is None else r2_settings.private_bucket_verified),
            key="r2_private_bucket_verified",
        )
        lifecycle_rule_verified = st.checkbox(
            "我已确认该存储桶已设置1日后自动删除对象",
            value=(False if r2_settings is None else r2_settings.lifecycle_rule_verified),
            key="r2_lifecycle_rule_verified",
        )
        if st.button(
            "保存R2非敏感配置",
            key="save_r2_coordinates",
            use_container_width=True,
        ):
            _run_action(
                lambda: _save_r2_coordinates(
                    account_id,
                    bucket_name,
                    signed_url_ttl_seconds,
                    private_bucket_verified,
                    lifecycle_rule_verified,
                ),
                "R2非敏感配置已保存；未满足全部安全条件时仍只发文字。",
            )

        access_key_id = st.text_input(
            "R2 Access Key ID",
            type="password",
            key="r2_access_key_id_input",
            help="只保存到macOS钥匙串，请勿发送到聊天、截图或GitHub。",
        )
        secret_access_key = st.text_input(
            "R2 Secret Access Key",
            type="password",
            key="r2_secret_access_key_input",
            help="只保存到macOS钥匙串，页面不会回显已存密钥。",
        )
        if st.button(
            "保存R2两项密钥",
            key="save_r2_credentials",
            use_container_width=True,
        ):
            _run_action(
                lambda: _save_r2_credentials(access_key_id, secret_access_key),
                "R2两项密钥已保存。",
            )

        st.info(
            "持仓文字摘要授权 ≠ 持仓K线图授权。即使文字摘要已允许外发，"
            "图片仍默认不外发。后续若分别授权Server酱或Bark发图，"
            "对应服务商也会接触短期签名HTTPS地址，并可能在到期前拉取或缓存图片。"
        )

    st.info(
        "晚报会分别尝试每个已配置通道；Server酱与Bark彼此独立，至少一个服务商受理后"
        "才登记去重状态。服务商受理不等于终端送达确认。"
    )
    st.warning("通知只提交衍生行动摘要。CSMAR原始数据、API凭据和完整持仓金额不会提交给第三方通道。")


render()
