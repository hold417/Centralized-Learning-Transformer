#!/bin/bash

# 訓練啟動腳本
# 功能：直接使用config.yaml啟動訓練

# 設置腳本選項
set -e  # 遇到錯誤時退出

# 默認參數
CONFIG_FILE="config.yaml"
RESUME_FILE=""
LOG_FILE="training.log"

# 顯示幫助信息
show_help() {
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -c, --config FILE       配置文件路徑 (默認: config.yaml)"
    echo "  -r, --resume FILE       從檢查點恢復訓練"
    echo "  -l, --log FILE          日誌文件名 (默認: training.log)"
    echo "  -h, --help              顯示此幫助信息"
    echo ""
    echo "示例:"
    echo "  $0                      # 默認配置啟動"
    echo "  $0 -c my_config.yaml   # 使用自定義配置"
    echo "  $0 -r checkpoint.pth   # 從檢查點恢復"
}

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_FILE="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
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

# 檢查必要文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "錯誤: 配置文件 '$CONFIG_FILE' 不存在"
    exit 1
fi

if [[ -n "$RESUME_FILE" && ! -f "$RESUME_FILE" ]]; then
    echo "錯誤: 恢復文件 '$RESUME_FILE' 不存在"
    exit 1
fi

# 構建訓練命令
TRAIN_CMD="python3 train.py --config $CONFIG_FILE"

if [[ -n "$RESUME_FILE" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_FILE"
fi

# 創建PID文件用於停止腳本
PID_FILE="training.pid"

# 顯示啟動信息
echo "========================================"
echo "啟動Transformer訓練"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "日誌文件: $LOG_FILE"
if [[ -n "$RESUME_FILE" ]]; then
    echo "恢復文件: $RESUME_FILE"
fi
echo "========================================"

# 啟動訓練（後台模式）
echo "以後台模式啟動訓練..."
nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"
echo "訓練已在後台啟動 (PID: $TRAIN_PID)"
echo "日誌文件: $LOG_FILE"
echo "停止訓練: ./stop.sh"
echo "監控訓練: tail -f $LOG_FILE"

# 等待一下確保進程啟動
sleep 2

# 檢查進程是否成功啟動
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "訓練進程運行正常"
else
    echo "警告: 訓練進程可能啟動失敗"
    rm -f "$PID_FILE"
fi