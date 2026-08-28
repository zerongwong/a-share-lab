#!/bin/bash

set -euo pipefail

LABEL="com.zerong.asharelab.daily-sync"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"

if /bin/launchctl print "$SERVICE" >/dev/null 2>&1; then
    /bin/launchctl bootout "$SERVICE"
fi
rm -f "$TARGET"

echo "每日收盘同步LaunchAgent已卸载。"
echo "本地行情、研究结果、日志和macOS钥匙串密钥均未删除。"
