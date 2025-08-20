#!/bin/bash
# DataLoader 데드락 감지 모니터링 스크립트

LOG_FILE="${1:-/tmp/pillsnap_training.log}"
TIMEOUT_SECONDS=${2:-300}  # 5분 기본 타임아웃
CHECK_INTERVAL=30  # 30초마다 체크

echo "🔍 DataLoader 데드락 모니터링 시작"
echo "로그 파일: $LOG_FILE"
echo "타임아웃: ${TIMEOUT_SECONDS}초"
echo "=================================="

# 마지막 배치 로그 시간 추적
last_batch_time=""
stuck_count=0

while true; do
    # 학습 프로세스가 실행 중인지 확인
    if ! pgrep -f "train_classification_stage" > /dev/null; then
        echo "✅ 학습 프로세스 종료됨"
        break
    fi
    
    # 최근 배치 로그 확인
    if [ -f "$LOG_FILE" ]; then
        current_batch=$(tail -20 "$LOG_FILE" | grep -E "Batch [0-9]+/[0-9]+:" | tail -1)
        current_time=$(date +%s)
        
        if [ -n "$current_batch" ]; then
            if [ "$current_batch" != "$last_batch_time" ]; then
                # 새로운 배치 진행됨
                echo "$(date '+%H:%M:%S') ✅ $current_batch"
                last_batch_time="$current_batch"
                stuck_count=0
            else
                # 같은 배치에서 멈춤
                stuck_count=$((stuck_count + CHECK_INTERVAL))
                echo "$(date '+%H:%M:%S') ⏳ 동일 배치 ${stuck_count}초 경과"
                
                if [ $stuck_count -ge $TIMEOUT_SECONDS ]; then
                    echo "❌ 데드락 감지! ${TIMEOUT_SECONDS}초 동안 진행 없음"
                    echo "마지막 배치: $current_batch"
                    
                    # 프로세스 상태 확인
                    echo "\n🔍 프로세스 상태:"
                    ps aux | grep "train_classification_stage" | grep -v grep
                    
                    # 스택 트레이스 (가능한 경우)
                    echo "\n🔍 스택 트레이스 시도:"
                    pkill -QUIT -f "train_classification_stage" 2>/dev/null || echo "스택 트레이스 실패"
                    
                    exit 1
                fi
            fi
        else
            # 배치 로그가 없으면 초기 단계
            epoch_log=$(tail -10 "$LOG_FILE" | grep -E "Epoch [0-9]+/[0-9]+" | tail -1)
            if [ -n "$epoch_log" ]; then
                stuck_count=$((stuck_count + CHECK_INTERVAL))
                echo "$(date '+%H:%M:%S') ⏳ 첫 배치 대기 중 ${stuck_count}초"
                
                if [ $stuck_count -ge $TIMEOUT_SECONDS ]; then
                    echo "❌ 첫 배치 데드락 감지!"
                    echo "마지막 에포크 로그: $epoch_log"
                    exit 1
                fi
            else
                echo "$(date '+%H:%M:%S') 🚀 초기화 중..."
            fi
        fi
    else
        echo "$(date '+%H:%M:%S') ⏳ 로그 파일 대기중..."
    fi
    
    sleep $CHECK_INTERVAL
done

echo "✅ 모니터링 완료"