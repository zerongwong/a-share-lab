from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE = PROJECT_ROOT / "src" / "ashare_lab" / "ui" / "pages" / "07_通知设置.py"


def test_notification_page_keeps_credentials_masked_and_local() -> None:
    source = PAGE.read_text(encoding="utf-8")

    assert source.count('type="password"') == 2
    assert "save_serverchan_sendkey" in source
    assert "save_bark_device_key" in source
    assert "macOS钥匙串" in source
    assert "发送微信测试" in source
    assert "Server酱 Turbo SendKey或官方推送地址" in source
    assert "发送iPhone测试" in source
    assert "把复制出的完整" in source
    assert "钥匙串暂时无法读取" in source
    assert "不会自动下单" in source
    assert "Server酱服务商已受理测试消息；微信终端送达未确认" in source
    assert "Bark服务商已受理测试消息；iPhone终端送达未确认" in source
    assert "至少一个服务商受理后" in source
    assert "才登记去重状态" in source
    assert "推送日志" in source
    assert "关注或绑定状态" in source
    assert "独立兜底" in source
    assert "测试消息已发送" not in source
    assert "os.environ" not in source
    assert "st.write(sendkey)" not in source
    assert "st.write(device_key)" not in source
