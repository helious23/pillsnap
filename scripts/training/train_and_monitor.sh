#!/bin/bash
# Stage 1 학습 + 통합 실시간 모니터링 스크립트

echo "🚀 PillSnap Stage 1 학습 + 모니터링 시작"
echo "=============================================="

# 로그 디렉토리 생성
LOG_DIR="/tmp/pillsnap_training"
mkdir -p "$LOG_DIR"

# 백그라운드로 학습 시작
echo "📚 백그라운드 학습 시작..."
nohup ./scripts/core/python_safe.sh -m src.training.train_classification_stage --stage 1 --epochs 30 --batch-size 112 > "$LOG_DIR/training.log" 2>&1 &
TRAINING_PID=$!

echo "🔍 학습 PID: $TRAINING_PID"
echo "📝 로그 파일: $LOG_DIR/training.log"

# 잠시 대기 (학습이 시작될 때까지)
sleep 3

# 통합 모니터링 시작
echo ""
echo "🖥️  실시간 모니터링 시작..."
echo "=============================================="

multitail \
  -l "bash -c 'while true; do 
    echo \"[$(date \"+%H:%M:%S\")] 🔥 TRAINING LOG\"; 
    tail -10 $LOG_DIR/training.log 2>/dev/null | grep -E \"(Epoch|Batch|Loss|Acc|목표|달성|완료)\" | tail -5;
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
      ps aux | grep train_classification_stage | grep -v grep | head -1 | awk \"{printf \"CPU: %s%% | MEM: %s%%\\n\", \$3, \$4}\";
    else
      echo \"❌ 학습 프로세스 종료됨\";
      echo \"📊 최종 로그 확인:\";
      tail -5 $LOG_DIR/training.log 2>/dev/null | grep -E \"(완료|달성|에러|error)\";
    fi;
    echo; 
    sleep 4; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[$(date \"+%H:%M:%S\")] 📈 PROGRESS SUMMARY\"; 
    # 최신 정확도 추출
    latest_acc=\$(tail -50 $LOG_DIR/training.log 2>/dev/null | grep -oE \"Acc=[0-9]+\.[0-9]+%\" | tail -1 | cut -d\"=\" -f2);
    if [ -n \"\$latest_acc\" ]; then
      echo \"현재 정확도: \$latest_acc (목표: 40.0%)\";
    else
      echo \"정확도 정보 로딩 중...\";
    fi;
    
    # 에포크 진행상황
    current_epoch=\$(tail -20 $LOG_DIR/training.log 2>/dev/null | grep -oE \"Epoch [0-9]+\" | tail -1 | cut -d\" \" -f2);
    if [ -n \"\$current_epoch\" ]; then
      echo \"현재 에포크: \$current_epoch/30\";
    fi;
    
    # 목표 달성 체크
    if tail -20 $LOG_DIR/training.log 2>/dev/null | grep -q \"목표.*달성\"; then
      echo \"🎉 목표 달성 완료!\";
    fi;
    
    echo; 
    sleep 5; 
  done'"

echo ""
echo "💡 팁:"
echo "   - Ctrl+C로 모니터링 종료"
echo "   - 학습은 백그라운드에서 계속 진행됩니다"
echo "   - 전체 로그: tail -f $LOG_DIR/training.log"