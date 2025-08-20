#!/bin/bash
# 간단한 Stage 2 훈련 모니터링

echo "🔍 Stage 2 훈련 모니터링 시작"
echo "==================================="

# 실행 중인 프로세스 찾기
TRAINING_PID=$(ps aux | grep "train_classification_stage" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAINING_PID" ]; then
    echo "❌ 실행 중인 훈련 프로세스를 찾을 수 없습니다."
    exit 1
fi

echo "✅ 훈련 프로세스: PID $TRAINING_PID"

while true; do
    clear
    echo "🚀 PillSnap Stage 2 실시간 모니터링"
    echo "=================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 프로세스 상태
    if ps -p $TRAINING_PID > /dev/null 2>&1; then
        echo "✅ 학습 진행 중 (PID: $TRAINING_PID)"
        ps aux | grep train_classification_stage | grep -v grep | head -1 | awk '{printf "CPU: %s%% | MEM: %s%% | 실행시간: %s\n", $3, $4, $10}'
    else
        echo "❌ 학습 프로세스 종료됨"
        break
    fi
    
    echo ""
    
    # GPU 상태 간단히
    echo "💻 GPU 상태:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | awk -F',' '{printf "   GPU: %s%% | 메모리: %s/%sMB | 온도: %s°C\n", $1, $2, $3, $4}'
    
    echo ""
    
    # 메인 로그 파일 고정
    LOG_FILE="/home/max16/ssd_pillsnap/exp/exp01/logs/__main___20250820.log"
    
    if [ -n "$LOG_FILE" ]; then
        echo "📝 최근 로그 ($LOG_FILE):"
        tail -10 "$LOG_FILE" | grep -E "(INFO|WARNING|ERROR)" | tail -3
        echo ""
        
        # 정확도 추출 (형식: Acc: XX.X% 또는 정확도: XX.X%)
        latest_acc=$(tail -50 "$LOG_FILE" 2>/dev/null | grep -oE "(Acc:|정확도:) [0-9]+\.[0-9]+%" | tail -1 | grep -oE "[0-9]+\.[0-9]+%")
        if [ -n "$latest_acc" ]; then
            echo "🎯 현재 정확도: $latest_acc (목표: 82.0%)"
        else
            echo "🎯 정확도: 아직 계산 중... (torch.compile 최적화 진행 중)"
        fi
        
        # 에포크 정보
        current_epoch=$(tail -20 "$LOG_FILE" 2>/dev/null | grep -oE "Epoch [0-9]+/[0-9]+" | tail -1)
        if [ -n "$current_epoch" ]; then
            echo "📊 현재 진행: $current_epoch"
        fi
        
        # 마지막 로그 시간
        last_log_time=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | tail -1)
        if [ -n "$last_log_time" ]; then
            echo "⏰ 마지막 로그: $last_log_time"
        fi
    else
        echo "📝 로그 파일을 찾을 수 없습니다."
    fi
    
    echo ""
    echo "Stage 2: 237개 클래스, ~18,960개 훈련 샘플"
    echo ""
    echo "Ctrl+C로 종료 (5초마다 갱신)"
    
    sleep 5
done

echo "모니터링 종료"
