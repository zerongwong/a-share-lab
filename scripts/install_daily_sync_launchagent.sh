#!/bin/bash

set -euo pipefail

LABEL="com.zerong.asharelab.daily-sync"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
TEMPLATE="$PROJECT_ROOT/config/$LABEL.plist.template"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET="$TARGET_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/A股研究助手"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"

# launchd can briefly retain the old service after bootout returns. Retry the
# same exact label; never escalate to root or restart another app/service.
bootstrap_with_retry() {
    for attempt in 1 2 3 4 5; do
        if /bin/launchctl bootstrap "$DOMAIN" "$TARGET"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "未找到项目虚拟环境：$PYTHON_BIN" >&2
    echo "请先在当前项目完成依赖安装，再重新执行。" >&2
    exit 2
fi
if [[ "$EUID" -eq 0 ]]; then
    echo "请使用当前macOS登录用户安装，不要使用sudo或root。" >&2
    exit 2
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "LaunchAgent模板不存在：$TEMPLATE" >&2
    exit 2
fi

mkdir -p "$TARGET_DIR" "$LOG_DIR"
chmod 700 "$LOG_DIR"

TEMP_PLIST="$(mktemp "${TMPDIR:-/tmp}/ashare-daily-sync.XXXXXX.plist")"
BACKUP_PLIST="$(mktemp "${TMPDIR:-/tmp}/ashare-daily-sync-backup.XXXXXX.plist")"
HAD_TARGET=false
WAS_LOADED=false
cleanup() {
    rm -f "$TEMP_PLIST" "$BACKUP_PLIST"
}
trap cleanup EXIT

if ! "$PYTHON_BIN" -c "import ashare_lab.cli.scheduled_sync"; then
    echo "调度模块无法从当前虚拟环境导入，请先重新安装项目依赖。" >&2
    exit 2
fi

"$PYTHON_BIN" -c \
    "import sys; from ashare_lab.cli.scheduled_sync import render_launchagent_plist; render_launchagent_plist(*sys.argv[1:])" \
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
/usr/bin/install -m 600 "$TEMP_PLIST" "$TARGET"
/bin/launchctl enable "$SERVICE"
if ! bootstrap_with_retry; then
    echo "新任务加载失败，正在恢复安装前状态。" >&2
    rm -f "$TARGET"
    if [[ "$HAD_TARGET" == true ]]; then
        /usr/bin/install -m 600 "$BACKUP_PLIST" "$TARGET"
        if [[ "$WAS_LOADED" == true ]]; then
            bootstrap_with_retry || true
        fi
    fi
    exit 2
fi
if ! /bin/launchctl print "$SERVICE" >/dev/null 2>&1; then
    echo "任务文件已写入但launchd没有登记该服务，请运行卸载脚本后重试。" >&2
    exit 2
fi

echo "每日收盘同步已安装并登记：15:30首次同步，16:30、18:30、19:30复核，20:20、20:50晚报前预检。"
echo "它与本地网页服务相互独立；不会连接券商或自动下单。"
echo "首次RunAtLoad结果请查看：$LOG_DIR/daily-sync.jsonl"
echo "查看状态：launchctl print $SERVICE"
