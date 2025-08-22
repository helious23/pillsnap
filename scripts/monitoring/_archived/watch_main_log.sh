#!/bin/bash
# 메인 로그 실시간 모니터링

LOG_FILE="/home/max16/ssd_pillsnap/exp/exp01/logs/__main___20250820.log"

echo "🔍 메인 로그 실시간 모니터링"
echo "파일: $LOG_FILE"
echo "==============================================="

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 로그 파일이 없습니다: $LOG_FILE"
    exit 1
fi

# 컬러 하이라이팅과 함께 실시간 출력
tail -f "$LOG_FILE" | while read line; do
    # 타임스탬프 추출
    timestamp=$(echo "$line" | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")
    
    # 중요한 로그만 컬러 출력
    if echo "$line" | grep -qE "(SUCCESS|✅|🎉)"; then
        echo -e "\033[32m$line\033[0m"  # 초록색
    elif echo "$line" | grep -qE "(ERROR|❌|실패)"; then
        echo -e "\033[31m$line\033[0m"  # 빨간색
    elif echo "$line" | grep -qE "(WARNING|⚠️)"; then
        echo -e "\033[33m$line\033[0m"  # 노란색
    elif echo "$line" | grep -qE "(Batch.*Loss=|Batch.*Acc=)"; then
        echo -e "\033[36m$line\033[0m"  # 청록색
    elif echo "$line" | grep -qE "(Epoch [0-9]+|📊|🚀|STEP)"; then
        echo -e "\033[35m$line\033[0m"  # 자주색
    else
        echo "$line"  # 기본색
    fi
done
