#!/bin/bash

# Detection 학습 진행에 집중한 리허설 스크립트
# 2025-08-25 개선된 Detection 디버깅 버전

set -e

echo "======================================"
echo "🚀 Stage 3 Two-Stage Pipeline 리허설 (Detection Focus)"
echo "======================================"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/max16/pillsnap:$PYTHONPATH
export PILLSNAP_DATA_ROOT=/home/max16/pillsnap_data

# 로그 디렉토리
LOG_DIR="/home/max16/pillsnap/artifacts/logs"
mkdir -p $LOG_DIR

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/rehearsal_detection_${TIMESTAMP}.log"

echo "📁 로그 파일: $LOG_FILE"
echo ""

# Detection state 초기화 확인
echo "🔍 Detection State 확인:"
if [ -f "/home/max16/pillsnap/artifacts/yolo/stage3/state.json" ]; then
    echo "현재 Detection State:"
    python -c "
import json
with open('/home/max16/pillsnap/artifacts/yolo/stage3/state.json', 'r') as f:
    state = json.load(f)
    print(f'  - det_epochs_done: {state.get(\"det_epochs_done\", 0)}')
    print(f'  - last_updated: {state.get(\"last_updated\", \"N/A\")}')
"
fi
echo ""

# 실행 명령
CMD="/home/max16/pillsnap/.venv/bin/python -m src.training.train_stage3_two_stage \
    --manifest-train /home/max16/pillsnap/artifacts/stage3/manifest_train.remove.csv \
    --manifest-val /home/max16/pillsnap/artifacts/stage3/manifest_val.remove.csv \
    --epochs 3 \
    --batch-size 8 \
    --lr-classifier 5e-5 \
    --lr-detector 1e-3 \
    --reset-best"

echo "🎯 실행 명령:"
echo "$CMD"
echo ""
echo "Detection 학습 진행 상황에 집중해서 모니터링합니다..."
echo "======================================"
echo ""

# 백그라운드 실행
$CMD > $LOG_FILE 2>&1

echo ""
echo "✅ 리허설 완료!"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
