#!/bin/bash
# 간단한 학습 모니터링

LOG_FILE="/tmp/pillsnap_training.log"

while true; do
    clear
    echo "🚀 PillSnap Stage 1 학습 모니터링 - $(date '+%H:%M:%S')"
    echo "=============================================="
    
    # GPU 상태
    echo "💻 GPU 상태:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | awk -F',' '{printf "   GPU 사용률: %s%% | 메모리: %s/%sMB | 온도: %s°C\n", $1, $2, $3, $4}'
    echo
    
    # 프로세스 상태
    echo "🔄 프로세스 상태:"
    if ps aux | grep -q "train_classification_stage.*30.*112"; then
        echo "   ✅ 학습 진행 중"
        ps aux | grep "train_classification_stage.*30.*112" | grep -v grep | head -1 | awk '{printf "   CPU: %s%% | 메모리: %s%%\n", $3, $4}'
    else
        echo "   ❌ 학습 프로세스 없음"
    fi
    echo
    
    # 최신 학습 로그 (더 자세히)
    echo "🔥 최신 학습 로그 (마지막 20줄):"
    if [ -f "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE" | sed 's/^/   /'
        echo
        
        # 현재 정확도 추출
        latest_acc=$(tail -20 "$LOG_FILE" | grep -oE "Acc=[0-9]+\.[0-9]+%" | tail -1 | cut -d'=' -f2)
        current_epoch=$(tail -10 "$LOG_FILE" | grep -oE "Epoch [0-9]+" | tail -1 | cut -d' ' -f2)
        
        if [ -n "$latest_acc" ]; then
            echo "📈 현재 정확도: $latest_acc (목표: 40.0%)"
        fi
        
        if [ -n "$current_epoch" ]; then
            echo "📊 진행도: $current_epoch/30 에포크"
        fi
        
        # 목표 달성 체크
        if tail -5 "$LOG_FILE" | grep -q "목표.*달성"; then
            echo "🎉 목표 달성 완료!"
        fi
    else
        echo "   로그 파일을 찾을 수 없습니다: $LOG_FILE"
    fi
    
    echo
    echo "=============================================="
    echo "💡 Ctrl+C로 모니터링 종료 | 전체 로그: tail -f $LOG_FILE"
    
    sleep 3
done