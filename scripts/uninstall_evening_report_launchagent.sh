#!/bin/bash

set -euo pipefail

LABEL="com.zerong.asharelab.evening-report"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"

if [[ "$EUID" -eq 0 ]]; then
    echo "请使用当前macOS登录用户卸载，不要使用sudo或root。" >&2
    exit 2
fi

/bin/launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
rm -f "$TARGET"

echo "21:00晚间报告任务已卸载。"
echo "研究数据、报告、日志、钥匙串密钥、网页服务和每日同步任务均未删除或修改。"
