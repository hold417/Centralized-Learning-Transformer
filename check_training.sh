#!/bin/bash

# 訓練狀態檢查腳本
# 功能：監控訓練進程和日誌狀態

# 設置腳本選項
set -e  # 遇到錯誤時退出

# 默認參數
PID_FILE="training.pid"
LOG_FILE="training.log"
WATCH_MODE=false
TAIL_LINES=50

# 顯示幫助信息
show_help() {
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -p, --pid-file FILE     PID文件路徑 (默認: training.pid)"
    echo "  -l, --log-file FILE     日誌文件路徑 (默認: training.log)"
    echo "  -w, --watch             監控模式 (實時更新)"
    echo "  -t, --tail LINES        顯示日誌尾部行數 (默認: 50)"
    echo "  -h, --help              顯示此幫助信息"
    echo ""
    echo "示例:"
    echo "  $0                      # 檢查訓練狀態"
    echo "  $0 -w                   # 監控模式"
    echo "  $0 -t 100               # 顯示最後100行日誌"
}

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pid-file)
            PID_FILE="$2"
            shift 2
            ;;
        -l|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        -t|--tail)
            TAIL_LINES="$2"
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

# 檢查進程狀態
check_process_status() {
    local status="未知"
    local pid=""
    local cpu_usage=""
    local memory_usage=""
    local runtime=""
    
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE")
        
        if kill -0 "$pid" 2>/dev/null; then
            status="運行中"
            
            # 獲取CPU和內存使用情況
            if command -v ps >/dev/null 2>&1; then
                local ps_info=$(ps -p "$pid" -o pid,ppid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null)
                if [[ -n "$ps_info" ]]; then
                    cpu_usage=$(echo "$ps_info" | awk '{print $3}')
                    memory_usage=$(echo "$ps_info" | awk '{print $4}')
                    runtime=$(echo "$ps_info" | awk '{print $5}')
                fi
            fi
        else
            status="已停止"
        fi
    else
        # 搜索可能的訓練進程
        local train_processes=$(ps aux | grep "python.*train.py" | grep -v grep | awk '{print $2}')
        if [[ -n "$train_processes" ]]; then
            status="運行中 (無PID文件)"
            pid="$train_processes"
        else
            status="未運行"
        fi
    fi
    
    echo "進程狀態: $status"
    if [[ -n "$pid" ]]; then
        echo "進程ID: $pid"
    fi
    if [[ -n "$cpu_usage" ]]; then
        echo "CPU使用率: ${cpu_usage}%"
    fi
    if [[ -n "$memory_usage" ]]; then
        echo "內存使用率: ${memory_usage}%"
    fi
    if [[ -n "$runtime" ]]; then
        echo "運行時間: $runtime"
    fi
}

# 檢查日誌狀態
check_log_status() {
    if [[ -f "$LOG_FILE" ]]; then
        local log_size=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
        local log_lines=$(wc -l < "$LOG_FILE" 2>/dev/null)
        local last_modified=$(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null)
        
        echo "日誌文件: $LOG_FILE"
        echo "文件大小: $log_size"
        echo "總行數: $log_lines"
        echo "最後修改: $last_modified"
        
        # 分析日誌內容
        if [[ -s "$LOG_FILE" ]]; then
            echo ""
            echo "=== 訓練進度分析 ==="
            
            # 查找最新的epoch信息
            local latest_epoch=$(grep -E "Epoch [0-9]+/" "$LOG_FILE" | tail -1 2>/dev/null)
            if [[ -n "$latest_epoch" ]]; then
                echo "最新Epoch: $latest_epoch"
            fi
            
            # 查找最佳驗證損失
            local best_loss=$(grep "保存最佳模型" "$LOG_FILE" | tail -1 2>/dev/null)
            if [[ -n "$best_loss" ]]; then
                echo "最佳模型: $best_loss"
            fi
            
            # 檢查是否有錯誤
            local error_count=$(grep -i "error\|exception\|traceback" "$LOG_FILE" | wc -l 2>/dev/null)
            if [[ $error_count -gt 0 ]]; then
                echo "錯誤數量: $error_count"
                echo "最近錯誤:"
                grep -i "error\|exception" "$LOG_FILE" | tail -3
            fi
            
            # 檢查是否完成
            if grep -q "訓練完成" "$LOG_FILE"; then
                echo "狀態: 訓練已完成"
            elif grep -q "早停觸發" "$LOG_FILE"; then
                echo "狀態: 早停觸發"
            else
                echo "狀態: 訓練中"
            fi
        fi
    else
        echo "日誌文件不存在: $LOG_FILE"
    fi
}

# 顯示最新日誌
show_recent_logs() {
    if [[ -f "$LOG_FILE" ]]; then
        echo ""
        echo "=== 最新日誌 (最後 $TAIL_LINES 行) ==="
        tail -n "$TAIL_LINES" "$LOG_FILE"
    fi
}

# 監控模式
watch_training() {
    echo "監控模式啟動 (按 Ctrl+C 退出)"
    echo "========================================"
    
    while true; do
        clear
        echo "時間: $(date)"
        echo "========================================"
        
        check_process_status
        echo ""
        check_log_status
        
        echo ""
        echo "=== 實時日誌 (最後 10 行) ==="
        if [[ -f "$LOG_FILE" ]]; then
            tail -n 10 "$LOG_FILE"
        fi
        
        echo ""
        echo "自動刷新中... (按 Ctrl+C 退出)"
        sleep 10
    done
}

# 主函數
main() {
    echo "========================================"
    echo "Transformer 訓練狀態檢查"
    echo "========================================"
    echo "時間: $(date)"
    echo "PID文件: $PID_FILE"
    echo "日誌文件: $LOG_FILE"
    echo "========================================"
    echo ""
    
    if [[ "$WATCH_MODE" == true ]]; then
        watch_training
    else
        check_process_status
        echo ""
        check_log_status
        show_recent_logs
    fi
}

# 設置中斷處理
trap 'echo ""; echo "檢查已中斷"; exit 0' INT TERM

# 運行主函數
main