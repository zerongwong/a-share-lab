from __future__ import annotations

import streamlit as st

from ashare_lab.domain.data_sources import DataAction, SourceRegistry, SourceStatus

STATUS_LABELS = {
    SourceStatus.NOT_CONNECTED: "🟡 未连接",
    SourceStatus.CONNECTED: "🟢 已连接",
    SourceStatus.BLOCKED_REQUIRE_WRITTEN_AUTHORIZATION: "🔴 等待书面授权",
    SourceStatus.EXPERIMENTAL: "🧪 实验数据源",
    SourceStatus.PRICE_BACKUP: "🟠 价格备用",
}

ACTION_LABELS = {
    DataAction.MARKET_DATA_READ: "读取行情",
    DataAction.MARKET_DATA_CACHE: "缓存行情",
    DataAction.FUNDAMENTAL_DATA_READ: "读取财务数据",
    DataAction.METADATA_READ: "读取元数据",
    DataAction.BODY_READ: "读取正文",
    DataAction.BODY_PERSIST: "正文落盘",
    DataAction.BODY_TO_LLM: "正文发送模型",
    DataAction.BODY_DISPLAY_UI: "正文界面展示",
    DataAction.BODY_EXPORT: "正文导出",
    DataAction.REDISTRIBUTE: "再分发",
}


def render() -> None:
    st.set_page_config(page_title="数据来源", page_icon="🔐", layout="wide")
    st.title("数据来源与授权")
    st.caption("每个结论都要知道数据来自哪里、何时可知，以及允许怎样使用。默认拒绝未授权用途。")

    st.info(
        "本页面只展示状态，不接收或保存任何账号凭据。购买授权后，请在系统钥匙串或提供商"
        "官方客户端中配置，再由本地适配器读取连接状态。"
    )
    st.warning(
        "财联社、证券时报正文目前处于阻断状态：没有可核验书面授权时，禁止落盘、发送给模型、"
        "在界面展示或导出。"
    )

    try:
        registry = SourceRegistry.load_default()
    except Exception as exc:
        st.error(f"数据来源配置无法读取：{exc}")
        return

    for source in registry.all():
        with st.container(border=True):
            heading, state = st.columns([3, 2])
            heading.subheader(source.display_name)
            state.markdown(f"**状态：{STATUS_LABELS[source.status]}**")
            st.write("用途：" + "；".join(source.purposes))

            if source.allowed_actions:
                rights = "、".join(
                    ACTION_LABELS[action]
                    for action in sorted(source.allowed_actions, key=lambda item: item.value)
                )
                st.write(f"配置中的许可动作：{rights}")
            else:
                st.write("配置中的许可动作：无（默认阻断）")

            if source.notes:
                st.caption(source.notes)
            official, application = st.columns(2)
            official.link_button("查看官方说明", source.official_url)
            application.link_button("查看申请／合作入口", source.application_url)

    st.caption("注意：来源状态和许可动作只代表本地安全门，不替代提供商合同与法律意见。")


render()
