#!/bin/bash

set -euo pipefail

LABEL="com.zerong.asharelab.evening-report"
DEADLINE_LABEL="com.zerong.asharelab.preopen-deadline"
PNL_LABEL="com.zerong.asharelab.holding-pnl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
TARGET_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
LABELS=("$LABEL" "$DEADLINE_LABEL" "$PNL_LABEL")
RENDERERS=("render_evening_report_launchagent_plist" "render_preopen_deadline_launchagent_plist" "render_holding_pnl_launchagent_plist")
HAD_TARGET=(false false false)
WAS_LOADED=(false false false)
WAS_DISABLED=(false false false)
TRANSACTION_STARTED=false
COMMITTED=false

if [[ "$EUID" -eq 0 ]]; then
    echo "请使用当前macOS登录用户安装，不要使用sudo或root。" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "未找到项目虚拟环境：$PYTHON_BIN" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c "import ashare_lab.cli.evening_report; import ashare_lab.cli.preopen_deadline; import ashare_lab.cli.holding_pnl"; then
    echo "盘前报告、独立截止提醒或持仓日报模块无法导入，请先完成项目更新与依赖安装。" >&2
    exit 2
fi
for CURRENT_LABEL in "${LABELS[@]}"; do
    if [[ ! -f "$PROJECT_ROOT/config/$CURRENT_LABEL.plist.template" ]]; then
        echo "LaunchAgent模板不存在：$CURRENT_LABEL" >&2
        exit 2
    fi
    if [[ -L "$TARGET_DIR/$CURRENT_LABEL.plist" ]]; then
        echo "拒绝覆盖符号链接任务文件：$CURRENT_LABEL" >&2
        exit 2
    fi
done

mkdir -p "$TARGET_DIR"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ashare-preopen-install.XXXXXX")"
restore_previous() {
    for INDEX in "${!LABELS[@]}"; do
        CURRENT_LABEL="${LABELS[$INDEX]}"
        TARGET="$TARGET_DIR/$CURRENT_LABEL.plist"
        SERVICE="$DOMAIN/$CURRENT_LABEL"
        /bin/launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
        if [[ "${HAD_TARGET[$INDEX]}" == true ]]; then
            /usr/bin/install -m 600 "$TEMP_DIR/$INDEX.previous.plist" "$TARGET" || {
                echo "任务文件恢复失败，请保留并检查备份：$TEMP_DIR/$INDEX.previous.plist" >&2
                return 1
            }
        else
            rm -f "$TARGET"
        fi
        /bin/launchctl enable "$SERVICE" || return 1
        if [[ "${WAS_LOADED[$INDEX]}" == true ]]; then
            /bin/launchctl bootstrap "$DOMAIN" "$TARGET" || return 1
        fi
        if [[ "${WAS_DISABLED[$INDEX]}" == true ]]; then
            /bin/launchctl disable "$SERVICE" || return 1
        fi
    done
}
cleanup() {
    EXIT_STATUS=$?
    if [[ "$TRANSACTION_STARTED" == true && "$COMMITTED" != true ]]; then
        echo "新盘前任务未能完成登记，正在恢复安装前状态。" >&2
        if ! restore_previous; then
            echo "恢复未完成，备份保留在：$TEMP_DIR" >&2
            exit 2
        fi
    fi
    for INDEX in "${!LABELS[@]}"; do
        rm -f "$TEMP_DIR/$INDEX.rendered.plist" "$TEMP_DIR/$INDEX.previous.plist"
    done
    rmdir "$TEMP_DIR"
    exit "$EXIT_STATUS"
}
trap cleanup EXIT

# Validate all definitions and save all old states BEFORE stopping any task.
for INDEX in "${!LABELS[@]}"; do
    CURRENT_LABEL="${LABELS[$INDEX]}"
    TARGET="$TARGET_DIR/$CURRENT_LABEL.plist"
    SERVICE="$DOMAIN/$CURRENT_LABEL"
    "$PYTHON_BIN" -c \
        "import sys; from ashare_lab.services import evening_report_launchagent as m; getattr(m, sys.argv[1])(*sys.argv[2:])" \
        "${RENDERERS[$INDEX]}" "$PROJECT_ROOT/config/$CURRENT_LABEL.plist.template" \
        "$TEMP_DIR/$INDEX.rendered.plist" "$PYTHON_BIN" "$PROJECT_ROOT"
    /usr/bin/plutil -lint "$TEMP_DIR/$INDEX.rendered.plist" >/dev/null
    if [[ -f "$TARGET" ]]; then
        cp "$TARGET" "$TEMP_DIR/$INDEX.previous.plist"
        chmod 600 "$TEMP_DIR/$INDEX.previous.plist"
        HAD_TARGET[$INDEX]=true
    fi
    if /bin/launchctl print "$SERVICE" >/dev/null 2>&1; then
        WAS_LOADED[$INDEX]=true
        if [[ "${HAD_TARGET[$INDEX]}" != true ]]; then
            echo "已有运行任务缺少可恢复的plist，安装停止：$CURRENT_LABEL" >&2
            exit 2
        fi
    fi
    WAS_DISABLED[$INDEX]="$(/bin/launchctl print-disabled "$DOMAIN" | "$PYTHON_BIN" -c \
        'import re, sys; print("true" if re.search(r"\"" + re.escape(sys.argv[1]) + r"\"\s*=>\s*true", sys.stdin.read()) else "false")' "$CURRENT_LABEL")"
done

TRANSACTION_STARTED=true
for INDEX in "${!LABELS[@]}"; do
    CURRENT_LABEL="${LABELS[$INDEX]}"
    TARGET="$TARGET_DIR/$CURRENT_LABEL.plist"
    SERVICE="$DOMAIN/$CURRENT_LABEL"
    if [[ "${WAS_LOADED[$INDEX]}" == true ]]; then
        /bin/launchctl bootout "$SERVICE"
    fi
    /usr/bin/install -m 600 "$TEMP_DIR/$INDEX.rendered.plist" "$TARGET"
    /bin/launchctl enable "$SERVICE"
    /bin/launchctl bootstrap "$DOMAIN" "$TARGET"
    /bin/launchctl print "$SERVICE" >/dev/null
done
COMMITTED=true

echo "周一至周五08:20盘前报告任务已登记，08:30、08:40、08:50重试；08:58关闭自动计算/提交窗口。"
echo "独立截止提醒已登记：08:58、08:59检查，不受主报告运行锁阻塞；两者均核验当天是否交易日。"
echo "持仓盈亏日报已登记：15:35、15:45、15:55有限尝试，仅15:30–16:00窗口并仅已登记持仓。"
echo "RunAtLoad可能立即唤起；各入口自行校验发送窗口和去重，不重启网页或每日数据同步。"
echo "查看状态：launchctl print $DOMAIN/$LABEL"
echo "查看截止提醒：launchctl print $DOMAIN/$DEADLINE_LABEL"
echo "查看持仓日报：launchctl print $DOMAIN/$PNL_LABEL"
