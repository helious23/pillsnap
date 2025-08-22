#!/bin/bash
# Stage 2 향상된 학습 모니터링 스크립트 - 실시간 로그 스트리밍

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 로그 파일 경로 (학습 시 리다이렉션되는 위치)
LOG_FILE="/tmp/pillsnap_stage2_training.log"

# 백그라운드 학습 시작 함수
start_stage2_training() {
    echo -e "${GREEN}🚀 Stage 2 학습 시작중...${NC}"
    
    # 이전 로그 백업
    if [ -f "$LOG_FILE" ]; then
        mv "$LOG_FILE" "${LOG_FILE}.$(date +%Y%m%d_%H%M%S).bak"
    fi
    
    # manifest 파일 확인
    manifest_file="/home/max16/pillsnap/artifacts/stage2/manifest_ssd.csv"
    if [ ! -f "$manifest_file" ]; then
        echo -e "${RED}❌ Stage 2 manifest 파일을 찾을 수 없습니다: $manifest_file${NC}"
        return 1
    fi
    
    samples=$(wc -l < "$manifest_file")
    samples=$((samples - 1))  # 헤더 제외
    echo -e "${CYAN}📊 Stage 2 데이터: ${samples}개 샘플${NC}"
    
    # 백그라운드로 학습 시작
    nohup ./scripts/core/python_safe.sh -m src.training.train_classification \
        --manifest "$manifest_file" \
        --stage 2 \
        --epochs "${1:-50}" \
        --batch-size "${2:-32}" \
        > "$LOG_FILE" 2>&1 &
    
    TRAINING_PID=$!
    echo -e "${GREEN}✅ Stage 2 학습 프로세스 시작됨 (PID: $TRAINING_PID)${NC}"
    echo -e "${YELLOW}📝 로그 파일: $LOG_FILE${NC}"
    sleep 2
}

# 실시간 모니터링 함수
monitor_stage2_training() {
    echo -e "${CYAN}📊 Stage 2 실시간 학습 모니터링 시작${NC}"
    echo "=================================================="
    
    # tail -f로 실시간 로그 추적
    if [ -f "$LOG_FILE" ]; then
        # 컬러 하이라이팅과 함께 출력
        tail -f "$LOG_FILE" | while IFS= read -r line; do
            # Batch 진행상황 하이라이팅
            if echo "$line" | grep -q "Batch.*Loss.*Acc"; then
                echo -e "${YELLOW}$line${NC}"
            # 검증 정확도 하이라이팅  
            elif echo "$line" | grep -q "Val Batch.*Acc"; then
                echo -e "${CYAN}$line${NC}"
            # Epoch 진행상황
            elif echo "$line" | grep -q "Epoch.*completed"; then
                echo -e "${BLUE}$line${NC}"
            # Stage 2 관련 정보
            elif echo "$line" | grep -q "Stage 2\|237.*클래스\|23700.*샘플"; then
                echo -e "${PURPLE}📋 $line${NC}"
            # 목표 달성 하이라이팅
            elif echo "$line" | grep -q "목표.*달성\|Target.*achieved"; then
                echo -e "${GREEN}🎉 $line${NC}"
            # 에러 하이라이팅
            elif echo "$line" | grep -qi "error\|fail\|exception"; then
                echo -e "${RED}❌ $line${NC}"
            # SUCCESS 하이라이팅
            elif echo "$line" | grep -q "SUCCESS\|완료"; then
                echo -e "${GREEN}✅ $line${NC}"
            # 중요 메트릭 하이라이팅
            elif echo "$line" | grep -q "최고 정확도\|best_accuracy\|accuracy.*%"; then
                echo -e "${PURPLE}⭐ $line${NC}"
            # Epoch 정보 하이라이팅
            elif echo "$line" | grep -q "Epoch [0-9]"; then
                echo -e "${BLUE}📈 $line${NC}"
            # 시스템 정보
            elif echo "$line" | grep -q "WSL 환경\|GPU\|Workers\|CUDA"; then
                echo -e "${WHITE}🖥️  $line${NC}"
            # Manifest 관련
            elif echo "$line" | grep -q "manifest\|Manifest"; then
                echo -e "${CYAN}📄 $line${NC}"
            else
                echo "$line"
            fi
        done
    else
        echo -e "${RED}❌ 로그 파일을 찾을 수 없습니다: $LOG_FILE${NC}"
        echo "Stage 2 학습을 먼저 시작하세요."
    fi
}

# 간단한 상태 확인 함수
check_stage2_status() {
    echo -e "${CYAN}💻 GPU 상태:${NC}"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits | head -1 | \
        awk -F',' '{printf "   GPU: %s%% | 메모리: %s/%sMB | 온도: %s°C\n", $1, $2, $3, $4}'
    
    echo -e "${CYAN}🔄 Stage 2 프로세스 상태:${NC}"
    if pgrep -f "train_classification.*stage.*2\|manifest_ssd.csv" > /dev/null; then
        echo -e "   ${GREEN}✅ Stage 2 학습 진행 중${NC}"
        ps aux | grep "train_classification\|manifest_ssd.csv" | grep -v grep | head -1 | \
            awk '{printf "   CPU: %s%% | 메모리: %s%%\n", $3, $4}'
    else
        echo -e "   ${RED}❌ Stage 2 학습 프로세스 없음${NC}"
    fi
    
    echo -e "${CYAN}📊 Stage 2 데이터 상태:${NC}"
    manifest_file="/home/max16/pillsnap/artifacts/stage2/manifest_ssd.csv"
    if [ -f "$manifest_file" ]; then
        samples=$(wc -l < "$manifest_file")
        samples=$((samples - 1))  # 헤더 제외
        echo -e "   ${GREEN}✅ Manifest: ${samples}개 샘플 준비됨${NC}"
    else
        echo -e "   ${RED}❌ Manifest 파일 없음${NC}"
    fi
    echo
}

# 메인 로직
case "${1:-monitor}" in
    start)
        # Stage 2 학습 시작 및 모니터링
        EPOCHS=${2:-50}
        BATCH_SIZE=${3:-32}
        
        echo -e "${GREEN}📚 PillSnap Stage 2 학습 시작${NC}"
        echo "Parameters: epochs=$EPOCHS, batch_size=$BATCH_SIZE"
        echo
        
        start_stage2_training $EPOCHS $BATCH_SIZE
        if [ $? -eq 0 ]; then
            monitor_stage2_training
        fi
        ;;
        
    monitor)
        # 기존 학습 모니터링만
        check_stage2_status
        monitor_stage2_training
        ;;
        
    status)
        # 상태만 확인
        check_stage2_status
        
        # 최근 로그 요약
        if [ -f "$LOG_FILE" ]; then
            echo -e "${CYAN}📝 최근 로그:${NC}"
            tail -10 "$LOG_FILE" | grep -E "Epoch|Batch.*Acc|목표|SUCCESS|Stage 2" || tail -5 "$LOG_FILE"
        fi
        ;;
        
    *)
        echo "사용법:"
        echo "  $0 start [epochs] [batch_size]    # Stage 2 학습 시작 및 모니터링"
        echo "  $0 monitor                        # 실행 중인 Stage 2 학습 모니터링"
        echo "  $0 status                         # 간단한 상태 확인"
        echo
        echo "예시:"
        echo "  $0 start 50 32      # Stage 2, 50 epochs, batch size 32"
        echo "  $0 monitor          # 실시간 로그 모니터링"
        echo
        echo -e "${YELLOW}💡 Stage 2 특징:${NC}"
        echo "  - 237개 클래스"
        echo "  - 23,700개 샘플"
        echo "  - SSD 기반 고속 데이터 로딩"
        ;;
esac