from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE = PROJECT_ROOT / "src" / "ashare_lab" / "ui" / "pages" / "07_通知设置.py"


def test_notification_page_keeps_credentials_masked_and_local() -> None:
    source = PAGE.read_text(encoding="utf-8")

    assert source.count('type="password"') == 4
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
    assert "Cloudflare R2" in source
    assert "save_chart_publisher_settings" in source
    assert "save_cloudflare_r2_access_key_id" in source
    assert "save_cloudflare_r2_secret_access_key" in source
    assert "保存本页不会联网、上传或发送图片" in source
    assert "我已确认该R2存储桶保持私有" in source
    assert "我已确认该存储桶已设置1日后自动删除对象" in source
    assert "存储桶必须保持私有" in source
    assert "1日自动删除" in source
    assert "两项都未确认时，晚报会自动退回纯文字" in source
    assert "R2对象目录（固定）" in source
    assert "对象目录固定为holding-charts" in source
    assert "持仓文字摘要授权 ≠ 持仓K线图授权" in source
    assert "Server酱或Bark发图" in source
    assert "短期签名HTTPS地址" in source
    assert "拉取或缓存图片" in source
    assert "os.environ" not in source
    assert "boto" not in source.lower()
    assert "测试消息已发送" not in source
    assert "st.write(sendkey)" not in source
    assert "st.write(device_key)" not in source
