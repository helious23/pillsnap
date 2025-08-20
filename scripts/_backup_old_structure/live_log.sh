#!/bin/bash
# 심플한 실시간 로그 뷰어

LOG_FILE="/tmp/pillsnap_training.log"

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}📊 PillSnap 학습 실시간 로그${NC}"
echo "=================================="
echo "로그 파일: $LOG_FILE"
echo "Ctrl+C로 종료"
echo "=================================="
echo

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}❌ 로그 파일이 없습니다.${NC}"
    echo "먼저 학습을 시작하세요:"
    echo "  ./scripts/python_safe.sh -m src.training.train_classification_stage --stage 1 --epochs 30 --batch-size 112 > $LOG_FILE 2>&1 &"
    exit 1
fi

# 실시간 로그 출력 (색상 강조)
tail -f "$LOG_FILE" | sed -u \
    -e "s/\(Batch.*Acc=[0-9.]*%\)/$(printf '\033[1;33m')\1$(printf '\033[0m')/g" \
    -e "s/\(Val Batch.*Acc=[0-9.]*%\)/$(printf '\033[0;36m')\1$(printf '\033[0m')/g" \
    -e "s/\(목표.*달성.*\)/$(printf '\033[0;32m')🎉 \1$(printf '\033[0m')/g" \
    -e "s/\(SUCCESS.*\)/$(printf '\033[0;32m')\1$(printf '\033[0m')/g" \
    -e "s/\(최고 정확도.*\)/$(printf '\033[0;35m')⭐ \1$(printf '\033[0m')/g" \
    -e "s/\(Epoch [0-9]*.*\)/$(printf '\033[0;34m')\1$(printf '\033[0m')/g"