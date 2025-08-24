#!/bin/bash
"""
Stage 3 Two-Stage 학습을 실시간 로그와 함께 실행

사용법:
  ./scripts/run_stage3_with_realtime_logs.sh [epochs] [batch_size] [port]

예제:
  ./scripts/run_stage3_with_realtime_logs.sh 30 20 8000
  
브라우저에서 http://localhost:8000 접속하여 실시간 로그 확인
"""

set -e

# 기본값
EPOCHS=${1:-30}
BATCH_SIZE=${2:-20} 
PORT=${3:-8000}

echo "🚀 Stage 3 Two-Stage 실시간 로그 학습 시작"
echo "  에포크: $EPOCHS"
echo "  배치 사이즈: $BATCH_SIZE" 
echo "  로그 포트: $PORT"
echo "  브라우저 접속: http://localhost:$PORT"
echo ""

cd /home/max16/pillsnap

# 가상환경 활성화
source .venv/bin/activate

# 실시간 로그 서버와 함께 학습 실행
python scripts/realtime_training_logger.py \
    --port $PORT \
    --command python -m src.training.train_stage3_two_stage --epochs $EPOCHS --batch-size $BATCH_SIZE