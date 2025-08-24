#!/bin/bash

# 데이터 품질 개선 통합 실행 스크립트
# Usage: ./run_all_fixes.sh [--dry-run|--execute]

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 디렉토리
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Python 환경 활성화
source /home/max16/pillsnap/.venv/bin/activate

# 실행 모드 확인
MODE="${1:---dry-run}"

if [ "$MODE" == "--dry-run" ]; then
    echo -e "${YELLOW}🔍 DRY RUN MODE - No actual changes will be made${NC}"
    DRY_RUN_FLAG="--dry-run"
    NO_DRY_RUN_FLAG=""
elif [ "$MODE" == "--execute" ]; then
    echo -e "${RED}⚠️  EXECUTE MODE - Changes will be applied!${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
    DRY_RUN_FLAG=""
    NO_DRY_RUN_FLAG="--no-dry-run"
else
    echo "Usage: $0 [--dry-run|--execute]"
    exit 1
fi

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/max16/pillsnap/artifacts/data_quality_reports/run_all_${TIMESTAMP}.log"

# 로그 함수
log() {
    echo -e "$1" | tee -a $LOG_FILE
}

# 단계별 실행 함수
run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3
    
    log "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    log "${GREEN}Step $step_num: $step_name${NC}"
    log "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    
    # 명령 실행
    if eval $command 2>&1 | tee -a $LOG_FILE; then
        log "${GREEN}✅ Step $step_num completed successfully${NC}"
        return 0
    else
        log "${RED}❌ Step $step_num failed${NC}"
        return 1
    fi
}

# 시작 메시지
log "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
log "${YELLOW}       DATA QUALITY IMPROVEMENT PIPELINE${NC}"
log "${YELLOW}       Started at: $(date)${NC}"
log "${YELLOW}═══════════════════════════════════════════════════════════${NC}"

# Step 0: 초기 품질 검사
log "\n${BLUE}[Initial Check]${NC}"
run_step 0 "Initial Quality Check" \
    "python final_quality_check.py"

# 각 단계별 최신 manifest 경로 추적
TRAIN_MANIFEST="artifacts/stage3/manifest_train.csv"
VAL_MANIFEST="artifacts/stage3/manifest_val.csv"

# Step 1: 손상 파일 정리 (최우선)
if run_step 1 "Clean Corrupted Files" \
    "python clean_corrupted_files.py $NO_DRY_RUN_FLAG"; then
    if [ "$MODE" == "--execute" ]; then
        TRAIN_MANIFEST="artifacts/stage3/manifest_train.cleaned.csv"
        VAL_MANIFEST="artifacts/stage3/manifest_val.cleaned.csv"
    fi
fi

# Step 2: Val-only 클래스 처리
if run_step 2 "Fix Val-only Classes" \
    "python fix_val_only_classes.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST --mode remove $NO_DRY_RUN_FLAG"; then
    if [ "$MODE" == "--execute" ]; then
        TRAIN_MANIFEST="artifacts/stage3/manifest_train.remove.csv"
        VAL_MANIFEST="artifacts/stage3/manifest_val.remove.csv"
    fi
fi

# Step 3: Combination 비율 조정
if run_step 3 "Balance Combination Ratio" \
    "python balance_combination_ratio.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST --target-ratio 0.25 --strategy oversample $NO_DRY_RUN_FLAG"; then
    if [ "$MODE" == "--execute" ]; then
        TRAIN_MANIFEST="artifacts/stage3/manifest_train.balanced_oversample.csv"
        VAL_MANIFEST="artifacts/stage3/manifest_val.balanced_oversample.csv"
    fi
fi

# Step 4: 클래스 가중치 계산
run_step 4 "Calculate Class Weights" \
    "python calculate_class_weights.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST --method balanced $NO_DRY_RUN_FLAG"

# Step 5: 최종 품질 검증
log "\n${BLUE}[Final Verification]${NC}"
run_step 5 "Final Quality Check" \
    "python final_quality_check.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST"

# 완료 메시지
log "\n${YELLOW}═══════════════════════════════════════════════════════════${NC}"
log "${YELLOW}       PIPELINE COMPLETED${NC}"
log "${YELLOW}       Ended at: $(date)${NC}"
log "${YELLOW}═══════════════════════════════════════════════════════════${NC}"

# 최종 manifest 경로 출력
if [ "$MODE" == "--execute" ]; then
    log "\n${GREEN}📁 Final Manifest Files:${NC}"
    log "   Train: $TRAIN_MANIFEST"
    log "   Val: $VAL_MANIFEST"
    log "\n${GREEN}💡 Next Steps:${NC}"
    log "   1. Review the final quality check results"
    log "   2. Update your training script to use the new manifests:"
    log "      --train-manifest $TRAIN_MANIFEST"
    log "      --val-manifest $VAL_MANIFEST"
    log "   3. Apply the class weights from artifacts/data_quality_reports/"
else
    log "\n${YELLOW}This was a DRY RUN. To apply changes, run:${NC}"
    log "   $0 --execute"
fi

log "\n📄 Full log saved to: $LOG_FILE"