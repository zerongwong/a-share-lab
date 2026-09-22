#!/bin/bash

set -euo pipefail

LABEL="com.zerong.asharelab.evening-report"
DEADLINE_LABEL="com.zerong.asharelab.preopen-deadline"
PNL_LABEL="com.zerong.asharelab.holding-pnl"
TARGET_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

if [[ "$EUID" -eq 0 ]]; then
    echo "请使用当前macOS登录用户卸载，不要使用sudo或root。" >&2
    exit 2
fi

for CURRENT_LABEL in "$LABEL" "$DEADLINE_LABEL" "$PNL_LABEL"; do
    SERVICE="$DOMAIN/$CURRENT_LABEL"
    if /bin/launchctl print "$SERVICE" >/dev/null 2>&1; then
        /bin/launchctl bootout "$SERVICE"
    fi
    rm -f "$TARGET_DIR/$CURRENT_LABEL.plist"
done

echo "盘前报告、独立截止提醒和持仓盈亏日报三个任务已卸载；需要恢复时重新运行安装脚本。"
echo "研究数据、报告、日志、钥匙串密钥、网页服务和每日同步任务均未删除或修改。"
