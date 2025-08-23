#!/bin/bash
# Stage 3 Classification 전용 학습 실행 스크립트
# Option 1 전략: Single 95% + Combination 5%, Classification 성능 극대화

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 프로젝트 루트 및 환경 설정
PROJECT_ROOT="/home/max16/pillsnap"
DATA_ROOT="/home/max16/pillsnap_data"
VENV_PATH="$PROJECT_ROOT/.venv"

log_info "Stage 3 Classification 전용 학습 시작"
log_info "전략: Option 1 (Single 95% + Combination 5%)"

# 1. 환경 설정 확인
cd "$PROJECT_ROOT"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    log_error "가상환경을 찾을 수 없습니다: $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"
log_success "가상환경 활성화"

# 환경변수 설정
export PILLSNAP_DATA_ROOT="$DATA_ROOT"
export CUDA_LAUNCH_BLOCKING=0  # 성능 최적화
export TORCH_CUDNN_V8_API_ENABLED=1  # cuDNN 최적화
log_info "환경변수 설정 완료"

# 2. GPU 및 시스템 상태 확인
log_info "시스템 상태 확인..."

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi 명령어를 찾을 수 없습니다"
    exit 1
fi

# GPU 메모리 확인
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
if [[ $GPU_MEM -lt 12000 ]]; then
    log_warning "GPU 메모리 부족: ${GPU_MEM}MB < 12GB 권장"
else
    log_success "GPU 메모리 충분: ${GPU_MEM}MB"
fi

# 디스크 공간 확인 (Native Linux SSD)
DISK_AVAIL=$(df "$DATA_ROOT" | tail -1 | awk '{print $4}')
DISK_AVAIL_GB=$((DISK_AVAIL / 1024 / 1024))
if [[ $DISK_AVAIL_GB -lt 20 ]]; then
    log_error "디스크 공간 부족: ${DISK_AVAIL_GB}GB < 20GB 필요"
    exit 1
else
    log_success "디스크 공간 충분: ${DISK_AVAIL_GB}GB"
fi

# 3. Manifest 파일 생성 (필요시)
MANIFEST_DIR="$PROJECT_ROOT/artifacts/stage3"
TRAIN_MANIFEST="$MANIFEST_DIR/manifest_train.csv"
VAL_MANIFEST="$MANIFEST_DIR/manifest_val.csv"

if [[ ! -f "$TRAIN_MANIFEST" ]] || [[ ! -f "$VAL_MANIFEST" ]]; then
    log_info "Stage 3 Manifest 생성 중..."
    
    python -m src.data.create_stage3_manifest
    
    if [[ $? -ne 0 ]]; then
        log_error "Manifest 생성 실패"
        exit 1
    fi
    
    log_success "Manifest 생성 완료"
else
    log_info "기존 Manifest 사용: $TRAIN_MANIFEST, $VAL_MANIFEST"
fi

# Manifest 유효성 확인
if [[ ! -f "$TRAIN_MANIFEST" ]]; then
    log_error "Train manifest 파일이 존재하지 않습니다: $TRAIN_MANIFEST"
    exit 1
fi

if [[ ! -f "$VAL_MANIFEST" ]]; then
    log_error "Val manifest 파일이 존재하지 않습니다: $VAL_MANIFEST"
    exit 1
fi

# Manifest 통계 출력
TRAIN_COUNT=$(wc -l < "$TRAIN_MANIFEST")
VAL_COUNT=$(wc -l < "$VAL_MANIFEST")
log_info "데이터 현황: Train ${TRAIN_COUNT}개, Val ${VAL_COUNT}개"

# 4. 실험 디렉토리 설정
EXP_DIR="$DATA_ROOT/exp/exp01"
LOGS_DIR="$EXP_DIR/logs"
CKPT_DIR="$EXP_DIR/checkpoints"
TB_DIR="$EXP_DIR/tb"

mkdir -p "$LOGS_DIR" "$CKPT_DIR" "$TB_DIR"
log_success "실험 디렉토리 생성: $EXP_DIR"

# 5. 학습 시작
log_info "========================================="
log_info "Stage 3 Classification 학습 시작"
log_info "========================================="
log_info "목표: Classification Accuracy 85%"
log_info "모델: EfficientNetV2-L"
log_info "데이터: Single 95% + Combination 5%"
log_info "========================================="

TRAIN_LOG="$LOGS_DIR/stage3_train.log"
TRAIN_ERR="$LOGS_DIR/stage3_train.err"

# 백그라운드 모니터링 시작
(
    while true; do
        sleep 30
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU Status:" >> "$LOGS_DIR/gpu_monitor.log"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits >> "$LOGS_DIR/gpu_monitor.log"
    done
) &
MONITOR_PID=$!

# 학습 실행
python -m src.training.train_stage3_classification \
    --config config.yaml \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --device cuda \
    > "$TRAIN_LOG" 2> "$TRAIN_ERR"

TRAIN_EXIT_CODE=$?

# 모니터링 중지
kill $MONITOR_PID 2>/dev/null || true

# 6. 결과 확인 및 리포트
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    log_success "Stage 3 Classification 학습 완료!"
    
    # 최고 성능 추출
    if [[ -f "$TRAIN_LOG" ]]; then
        BEST_ACC=$(grep "최고 정확도" "$TRAIN_LOG" | tail -1 | grep -oP '\d+\.\d+(?=%)')
        BEST_F1=$(grep "최고 Macro F1" "$TRAIN_LOG" | tail -1 | grep -oP '\d+\.\d+')
        TARGET_ACHIEVED=$(grep "목표 달성" "$TRAIN_LOG" | tail -1 | grep -o "✅" || echo "❌")
        
        log_info "========================================="
        log_info "Stage 3 최종 결과"
        log_info "========================================="
        log_info "최고 정확도: ${BEST_ACC:-N/A}%"
        log_info "최고 Macro F1: ${BEST_F1:-N/A}"
        log_info "목표 달성 (85%): $TARGET_ACHIEVED"
        log_info "========================================="
        
        # 체크포인트 확인
        BEST_CKPT="$CKPT_DIR/stage3_classification_best.pt"
        if [[ -f "$BEST_CKPT" ]]; then
            CKPT_SIZE=$(du -h "$BEST_CKPT" | cut -f1)
            log_success "Best 체크포인트 저장: $BEST_CKPT (${CKPT_SIZE})"
        fi
    fi
    
    # 로그 파일 정보
    log_info "로그 파일:"
    log_info "  - 학습 로그: $TRAIN_LOG"
    log_info "  - 에러 로그: $TRAIN_ERR"
    log_info "  - GPU 모니터: $LOGS_DIR/gpu_monitor.log"
    
else
    log_error "Stage 3 Classification 학습 실패 (exit code: $TRAIN_EXIT_CODE)"
    
    # 에러 로그 확인
    if [[ -f "$TRAIN_ERR" ]] && [[ -s "$TRAIN_ERR" ]]; then
        log_error "에러 내용:"
        tail -20 "$TRAIN_ERR"
    fi
    
    exit $TRAIN_EXIT_CODE
fi

# 7. 다음 단계 안내
log_info "다음 단계:"
log_info "1. TensorBoard로 학습 과정 확인: tensorboard --logdir $TB_DIR"
log_info "2. Stage 4 준비 (Two-Stage 통합)를 위한 계획 수립"
log_info "3. 성능 분석 및 최적화 검토"

log_success "Stage 3 Classification 전용 학습 완료!"