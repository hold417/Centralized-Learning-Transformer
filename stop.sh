#!/bin/bash

# VPS訓練停止腳本
# 功能：安全停止訓練進程並保存當前狀態

# 設置腳本選項
set -e  # 遇到錯誤時退出

# 默認參數
PID_FILE="training.pid"
FORCE_KILL=false
WAIT_TIME=30

# 顯示幫助信息
show_help() {
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -p, --pid-file FILE     PID文件路徑 (默認: training.pid)"
    echo "  -f, --force             強制終止 (使用SIGKILL)"
    echo "  -w, --wait SECONDS      等待時間 (默認: 30秒)"
    echo "  -h, --help              顯示此幫助信息"
    echo ""
    echo "示例:"
    echo "  $0                      # 正常停止訓練"
    echo "  $0 -f                   # 強制停止"
    echo "  $0 -w 60                # 等待60秒後強制停止"
}

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pid-file)
            PID_FILE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_KILL=true
            shift
            ;;
        -w|--wait)
            WAIT_TIME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知選項: $1"
            show_help
            exit 1
            ;;
    esac
done

# 檢查是否有訓練進程運行
if [[ ! -f "$PID_FILE" ]]; then
    echo "未找到PID文件: $PID_FILE"
    echo "檢查是否有Python訓練進程在運行..."
    
    # 搜索可能的訓練進程
    TRAIN_PROCESSES=$(ps aux | grep "python.*train.py" | grep -v grep | awk '{print $2}')
    
    if [[ -z "$TRAIN_PROCESSES" ]]; then
        echo "沒有找到訓練進程"
        exit 0
    else
        echo "找到以下訓練進程:"
        ps aux | grep "python.*train.py" | grep -v grep
        echo ""
        echo "請選擇要停止的進程PID，或使用 -f 選項強制停止所有相關進程"
        exit 1
    fi
fi

# 讀取PID
TRAIN_PID=$(cat "$PID_FILE")

# 檢查進程是否存在
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "進程 $TRAIN_PID 不存在或已經停止"
    rm -f "$PID_FILE"
    exit 0
fi

echo "========================================"
echo "停止Transformer訓練"
echo "========================================"
echo "PID文件: $PID_FILE"
echo "訓練PID: $TRAIN_PID"
echo "強制終止: $FORCE_KILL"
echo "等待時間: $WAIT_TIME 秒"
echo "========================================"

# 顯示進程信息
echo "當前訓練進程信息:"
ps -p "$TRAIN_PID" -o pid,ppid,cmd || echo "無法獲取進程信息"
echo ""

# 停止進程的函數
stop_process() {
    local pid=$1
    local signal=$2
    local description=$3
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "發送 $description 信號到進程 $pid..."
        kill "$signal" "$pid"
        return 0
    else
        echo "進程 $pid 已經停止"
        return 1
    fi
}

# 等待進程停止的函數
wait_for_stop() {
    local pid=$1
    local timeout=$2
    local count=0
    
    while kill -0 "$pid" 2>/dev/null && [[ $count -lt $timeout ]]; do
        sleep 1
        count=$((count + 1))
        if [[ $((count % 5)) -eq 0 ]]; then
            echo "等待進程停止... ($count/$timeout 秒)"
        fi
    done
    
    if kill -0 "$pid" 2>/dev/null; then
        return 1  # 進程仍在運行
    else
        return 0  # 進程已停止
    fi
}

# 清理函數
cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        echo "清理PID文件: $PID_FILE"
        rm -f "$PID_FILE"
    fi
    
    # 清理臨時配置文件
    if [[ -f "vps_config.yaml" ]]; then
        echo "清理臨時配置文件: vps_config.yaml"
        rm -f "vps_config.yaml"
    fi
}

# 主停止邏輯
if [[ "$FORCE_KILL" == true ]]; then
    echo "強制終止進程..."
    stop_process "$TRAIN_PID" -KILL "SIGKILL"
    cleanup
    echo "訓練進程已強制終止"
else
    echo "正常停止訓練進程..."
    
    # 首先嘗試正常終止 (SIGTERM)
    if stop_process "$TRAIN_PID" -TERM "SIGTERM"; then
        echo "等待進程優雅停止..."
        
        if wait_for_stop "$TRAIN_PID" "$WAIT_TIME"; then
            echo "訓練進程已正常停止"
            cleanup
            exit 0
        else
            echo "等待超時，嘗試強制終止..."
            if stop_process "$TRAIN_PID" -KILL "SIGKILL"; then
                echo "訓練進程已強制終止"
                cleanup
                exit 0
            else
                echo "無法終止進程"
                exit 1
            fi
        fi
    else
        cleanup
        exit 0
    fi
fi

# 驗證進程是否已停止
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "警告: 進程可能仍在運行"
    exit 1
else
    echo "確認: 訓練進程已停止"
    cleanup
    exit 0
fi