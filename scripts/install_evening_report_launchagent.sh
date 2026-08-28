#!/bin/bash

set -euo pipefail

LABEL="com.zerong.asharelab.evening-report"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
TEMPLATE="$PROJECT_ROOT/config/$LABEL.plist.template"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET="$TARGET_DIR/$LABEL.plist"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"

if [[ "$EUID" -eq 0 ]]; then
    echo "请使用当前macOS登录用户安装，不要使用sudo或root。" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "未找到项目虚拟环境：$PYTHON_BIN" >&2
    echo "请先在当前项目完成依赖安装，再重新执行。" >&2
    exit 2
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "LaunchAgent模板不存在：$TEMPLATE" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c "import ashare_lab.cli.evening_report"; then
    echo "晚间报告模块无法从当前虚拟环境导入，请先完成项目更新与依赖安装。" >&2
    exit 2
fi

mkdir -p "$TARGET_DIR"
TEMP_PLIST="$(mktemp "${TMPDIR:-/tmp}/ashare-evening-report.XXXXXX.plist")"
BACKUP_PLIST="$(mktemp "${TMPDIR:-/tmp}/ashare-evening-report-backup.XXXXXX.plist")"
HAD_TARGET=false
WAS_LOADED=false
cleanup() {
    rm -f "$TEMP_PLIST" "$BACKUP_PLIST"
}
trap cleanup EXIT

"$PYTHON_BIN" -c \
    "import sys; from ashare_lab.services.evening_report_launchagent import render_evening_report_launchagent_plist; render_evening_report_launchagent_plist(*sys.argv[1:])" \
    "$TEMPLATE" "$TEMP_PLIST" "$PYTHON_BIN" "$PROJECT_ROOT"
/usr/bin/plutil -lint "$TEMP_PLIST" >/dev/null
chmod 600 "$TEMP_PLIST"

if [[ -f "$TARGET" ]]; then
    cp "$TARGET" "$BACKUP_PLIST"
    chmod 600 "$BACKUP_PLIST"
    HAD_TARGET=true
fi
if /bin/launchctl print "$SERVICE" >/dev/null 2>&1; then
    WAS_LOADED=true
    /bin/launchctl bootout "$SERVICE"
fi

restore_previous() {
    /bin/launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
    rm -f "$TARGET"
    if [[ "$HAD_TARGET" == true ]]; then
        /usr/bin/install -m 600 "$BACKUP_PLIST" "$TARGET"
        if [[ "$WAS_LOADED" == true ]]; then
            /bin/launchctl bootstrap "$DOMAIN" "$TARGET" || true
        fi
    fi
}

if ! /usr/bin/install -m 600 "$TEMP_PLIST" "$TARGET" \
    || ! /bin/launchctl enable "$SERVICE" \
    || ! /bin/launchctl bootstrap "$DOMAIN" "$TARGET" \
    || ! /bin/launchctl print "$SERVICE" >/dev/null 2>&1; then
    echo "新晚间报告任务未能完成登记，正在恢复安装前状态。" >&2
    restore_previous
    exit 2
fi

echo "周日至周四21:00晚间报告任务已安装并登记；周五、周六不做计划推送。"
echo "它与本地网页及每日数据同步任务使用不同label，不会重启或修改它们。"
echo "RunAtLoad可能立即唤起一次；报告服务还会执行周五/周六发送门、交易日链和输入版本去重。"
echo "查看状态：launchctl print $SERVICE"
