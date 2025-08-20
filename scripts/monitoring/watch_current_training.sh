#!/bin/bash
# 현재 실행 중인 훈련 프로세스 실시간 모니터링

echo "🔍 현재 실행 중인 훈련 프로세스 검색..."

# 현재 실행 중인 train_classification_stage 프로세스 찾기
TRAINING_PID=$(ps aux | grep "train_classification_stage" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAINING_PID" ]; then
    echo "❌ 실행 중인 훈련 프로세스를 찾을 수 없습니다."
    echo "다음 명령어로 훈련을 시작하세요:"
    echo "./scripts/monitoring/train_and_monitor_stage2.sh"
    exit 1
fi

echo "✅ 훈련 프로세스 발견: PID $TRAINING_PID"
echo "🖥️  실시간 모니터링 시작..."
echo "=============================================="

# 임시 로그 경로들 확인
LOG_PATHS=(
    "/tmp/pillsnap_training_stage2/training.log"
    "/tmp/pillsnap_training/training.log"
    "/tmp/training.log"
)

FOUND_LOG=""
for log_path in "${LOG_PATHS[@]}"; do
    if [ -f "$log_path" ]; then
        FOUND_LOG="$log_path"
        echo "📝 로그 파일 발견: $FOUND_LOG"
        break
    fi
done

if [ -z "$FOUND_LOG" ]; then
    echo "⚠️  로그 파일을 찾을 수 없어 프로세스 상태만 모니터링합니다."
    FOUND_LOG="/dev/null"
fi

multitail \
  -l "bash -c 'while true; do 
    echo \"[$(date \"+%H:%M:%S\")] 🔥 TRAINING LOG\"; 
    if [ \"$FOUND_LOG\" != \"/dev/null\" ]; then
      tail -10 $FOUND_LOG 2>/dev/null | grep -E \"(Epoch|Batch|Loss|Acc|목표|달성|완료|INFO)\" | tail -5;
    else
      echo \"로그 파일 없음 - 프로세스 상태만 확인\";
    fi;
    echo; 
    sleep 2; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[$(date \"+%H:%M:%S\")] 💻 GPU STATUS\"; 
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | awk -F\",\" \"{printf \"GPU: %s%% | MEM: %s/%sMB | TEMP: %s°C\\n\", \$1, \$2, \$3, \$4}\";
    echo; 
    sleep 3; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[$(date \"+%H:%M:%S\")] 🔄 PROCESS STATUS\"; 
    if ps -p $TRAINING_PID > /dev/null 2>&1; then
      echo \"✅ 학습 진행 중 (PID: $TRAINING_PID)\";
      ps aux | grep train_classification_stage | grep -v grep | head -1 | awk \"{printf \"CPU: %s%% | MEM: %s%% | TIME: %s\\n\", \$3, \$4, \$10}\";
    else
      echo \"❌ 학습 프로세스 종료됨\";
      if [ \"$FOUND_LOG\" != \"/dev/null\" ]; then
        echo \"📊 최종 로그 확인:\";
        tail -5 $FOUND_LOG 2>/dev/null | grep -E \"(완료|달성|에러|error)\";
      fi;
    fi;
    echo; 
    sleep 4; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[$(date \"+%H:%M:%S\")] 📈 PROGRESS SUMMARY\"; 
    
    if [ \"$FOUND_LOG\" != \"/dev/null\" ]; then
      # 최신 정확도 추출
      latest_acc=\$(tail -50 $FOUND_LOG 2>/dev/null | grep -oE \"Acc=[0-9]+\.[0-9]+%\" | tail -1 | cut -d\"=\" -f2);
      if [ -n \"\$latest_acc\" ]; then
        echo \"현재 정확도: \$latest_acc (목표: 40.0%)\";
      else
        echo \"정확도 정보 로딩 중...\";
      fi;
      
      # 에포크 진행상황
      current_epoch=\$(tail -20 $FOUND_LOG 2>/dev/null | grep -oE \"Epoch [0-9]+\" | tail -1 | cut -d\" \" -f2);
      if [ -n \"\$current_epoch\" ]; then
        echo \"현재 에포크: \$current_epoch\";
      fi;
    else
      echo \"로그 정보 없음\";
    fi;
    
    # Stage 2 고정 정보
    echo \"Stage: 2 (237개 클래스)\";
    echo \"예상 샘플: ~18,960개 훈련 | ~4,740개 검증\";
    
    # 실행 시간 계산
    if ps -p $TRAINING_PID > /dev/null 2>&1; then
      runtime=\$(ps -o etime= -p $TRAINING_PID | tr -d ' ');
      echo \"실행 시간: \$runtime\";
    fi;
    
    echo; 
    sleep 5; 
  done'"

echo ""
echo "💡 팁:"
echo "   - Ctrl+C로 모니터링 종료"
echo "   - 학습은 백그라운드에서 계속 진행됩니다"
if [ "$FOUND_LOG" != "/dev/null" ]; then
    echo "   - 전체 로그: tail -f $FOUND_LOG"
fi