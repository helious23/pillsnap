#!/bin/bash
# 학습 시작 + 실시간 모니터링 통합 스크립트

# 파라미터 (기본값 설정)
STAGE=${1:-1}
EPOCHS=${2:-30}
BATCH_SIZE=${3:-112}

# 색상
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

LOG_FILE="/tmp/pillsnap_training.log"

echo -e "${GREEN}🚀 PillSnap Stage $STAGE 학습 시작${NC}"
echo "=================================================="
echo -e "${CYAN}📊 Parameters:${NC}"
echo "   Stage: $STAGE"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Log File: $LOG_FILE"
echo "=================================================="
echo

# 이전 로그 백업
if [ -f "$LOG_FILE" ]; then
    BACKUP="${LOG_FILE}.$(date +%Y%m%d_%H%M%S).bak"
    mv "$LOG_FILE" "$BACKUP"
    echo -e "${YELLOW}📦 이전 로그 백업: $BACKUP${NC}"
fi

# 학습 명령어 실행 (tee로 동시에 파일 저장 + 화면 출력)
echo -e "${GREEN}🏋️ 학습 시작 및 실시간 모니터링...${NC}"
echo "=================================================="

export PILLSNAP_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"

# 실시간 출력과 로그 저장을 동시에
./scripts/python_safe.sh -m src.training.train_classification_stage \
    --stage $STAGE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    2>&1 | tee "$LOG_FILE" | while IFS= read -r line; do
        # Batch 진행상황 하이라이팅
        if echo "$line" | grep -q "Batch.*Loss.*Acc"; then
            echo -e "${YELLOW}$line${NC}"
        # 검증 정확도 하이라이팅  
        elif echo "$line" | grep -q "Val Batch.*Acc"; then
            echo -e "${CYAN}$line${NC}"
        # 목표 달성 하이라이팅
        elif echo "$line" | grep -q "목표.*달성\|목표 달성"; then
            echo -e "${GREEN}🎉 $line${NC}"
        # SUCCESS 하이라이팅
        elif echo "$line" | grep -q "SUCCESS"; then
            echo -e "${GREEN}✅ $line${NC}"
        # 최고 정확도 하이라이팅
        elif echo "$line" | grep -q "최고 정확도\|best_accuracy"; then
            echo -e "\033[0;35m⭐ $line${NC}"
        # Epoch 정보 하이라이팅
        elif echo "$line" | grep -q "Epoch [0-9]"; then
            echo -e "\033[0;34m$line${NC}"
        # 에러 하이라이팅
        elif echo "$line" | grep -qi "error\|fail"; then
            echo -e "\033[0;31m❌ $line${NC}"
        # 시스템 정보
        elif echo "$line" | grep -q "WSL 환경\|Workers"; then
            echo -e "\033[1;37m$line${NC}"
        else
            echo "$line"
        fi
    done

echo
echo "=================================================="
echo -e "${GREEN}✅ 학습 완료!${NC}"
echo -e "${CYAN}📝 전체 로그: $LOG_FILE${NC}"