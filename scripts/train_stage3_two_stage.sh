#!/bin/bash
"""
Stage 3 Two-Stage Pipeline 학습 실행 스크립트

YOLOv11x Detection + EfficientNetV2-L Classification 통합 학습
- 95% Single + 5% Combination 데이터
- RTX 5080 최적화 (Mixed Precision, torch.compile)
- 목표: Classification 85%, Detection mAP@0.5 30%
"""

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m' 
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
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

# 프로젝트 루트 경로
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_info "Stage 3 Two-Stage Pipeline 학습 시작"
log_info "프로젝트 경로: $PROJECT_ROOT"

# Python 가상환경 활성화
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    log_success "Python 가상환경 활성화됨"
else
    log_error "Python 가상환경을 찾을 수 없습니다: .venv/bin/activate"
    exit 1
fi

# 필수 파일 존재 확인
REQUIRED_FILES=(
    "config.yaml"
    "artifacts/stage3/manifest_train.csv"
    "artifacts/stage3/manifest_val.csv"
    "src/training/train_stage3_two_stage.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        log_error "필수 파일이 없습니다: $file"
        exit 1
    fi
done

log_success "모든 필수 파일 확인 완료"

# GPU 상태 확인
log_info "GPU 상태 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    
    # RTX 5080 확인
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX 5080"; then
        log_success "RTX 5080 GPU 감지됨"
    else
        log_warning "RTX 5080이 아닌 GPU가 감지됨. 성능이 다를 수 있습니다."
    fi
else
    log_warning "nvidia-smi를 찾을 수 없습니다. GPU 상태를 확인할 수 없습니다."
fi

# Stage 3 manifest 데이터 확인
log_info "Stage 3 manifest 데이터 확인 중..."
python3 -c "
import pandas as pd

# 훈련 데이터 확인
train_df = pd.read_csv('artifacts/stage3/manifest_train.csv')
val_df = pd.read_csv('artifacts/stage3/manifest_val.csv')

print(f'훈련 샘플: {len(train_df):,}개')
print(f'검증 샘플: {len(val_df):,}개')
print(f'총 클래스: {train_df[\"edi_code\"].nunique()}개')

# Single/Combination 비율 확인
train_single = train_df[train_df['pill_type'] == 'single']
train_combo = train_df[train_df['pill_type'] == 'combination']

single_ratio = len(train_single) / len(train_df)
combo_ratio = len(train_combo) / len(train_df)

print(f'Single: {len(train_single):,}개 ({single_ratio:.1%})')
print(f'Combination: {len(train_combo):,}개 ({combo_ratio:.1%})')

# Stage 3 목표와 비교
if single_ratio >= 0.90:
    print('✅ Single 비율 적절 (≥90%)')
else:
    print('❌ Single 비율 부족 (<90%)')

if combo_ratio >= 0.03:
    print('✅ Combination 비율 적절 (≥3%)')
else:
    print('⚠️ Combination 비율 부족 (Detection 학습에 제한적)')
"

# 메모리 및 스토리지 확인
log_info "시스템 리소스 확인 중..."
echo "RAM 사용량:"
free -h | head -2

echo -e "\n디스크 공간:"
df -h "$PROJECT_ROOT" | tail -1

# Stage 3 Two-Stage Pipeline 학습 실행
log_info "Stage 3 Two-Stage Pipeline 학습 시작..."

# 학습 시작 시간 기록
START_TIME=$(date +%s)

# Python 학습 스크립트 실행
python3 src/training/train_stage3_two_stage.py \
    --config config.yaml \
    --manifest-train artifacts/stage3/manifest_train.csv \
    --manifest-val artifacts/stage3/manifest_val.csv \
    --device cuda

# 학습 결과 확인
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_info "학습 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"

if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Stage 3 Two-Stage Pipeline 학습 완료!"
    
    # 결과 파일 확인
    if [[ -d "artifacts/stage3/checkpoints" ]]; then
        log_info "생성된 체크포인트:"
        ls -la artifacts/stage3/checkpoints/
    fi
    
    # GPU 메모리 상태 확인
    if command -v nvidia-smi &> /dev/null; then
        log_info "학습 완료 후 GPU 메모리 상태:"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    fi
    
    # 성능 요약
    log_info "Stage 3 목표 달성 여부:"
    echo "  - Classification 목표: ≥85% 정확도"
    echo "  - Detection 목표: ≥30% mAP@0.5"
    echo "  - 시간 제한: ≤16시간 (실제: ${HOURS}h ${MINUTES}m)"
    
    if [[ $HOURS -le 16 ]]; then
        log_success "시간 제한 내 학습 완료"
    else
        log_warning "시간 제한 초과"
    fi
    
else
    log_error "Stage 3 Two-Stage Pipeline 학습 실패 (종료 코드: $EXIT_CODE)"
    
    # 에러 로그 확인
    if [[ -d "artifacts/stage3/logs" ]]; then
        log_info "최근 에러 로그:"
        find artifacts/stage3/logs -name "*error*" -type f -mtime -1 | head -3 | xargs tail -10
    fi
    
    exit $EXIT_CODE
fi

# Stage 4 준비 상태 확인
log_info "Stage 4 준비 상태 확인..."

# Detection 기능 검증 완료 여부
if [[ -f "artifacts/stage3/checkpoints/stage3_detection_best.pt" ]]; then
    log_success "Detection 체크포인트 생성 완료 - Stage 4 Detection 준비됨"
else
    log_warning "Detection 체크포인트 누락 - Stage 4에서 Detection 초기화 필요"
fi

# Classification 성능 유지 여부  
if [[ -f "artifacts/stage3/checkpoints/stage3_classification_best.pt" ]]; then
    log_success "Classification 체크포인트 생성 완료 - Stage 4 Classification 준비됨"
else
    log_warning "Classification 체크포인트 누락"
fi

log_success "Stage 3 Two-Stage Pipeline 전체 프로세스 완료!"
log_info "다음 단계: Stage 4에서 Two-Stage Pipeline 프로덕션 학습"