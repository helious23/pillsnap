#!/bin/bash
set -euo pipefail

# Stage별 성능 평가 실행 스크립트
# Usage: ./evaluate_stage.sh [STAGE_NUMBER] [OPTIONS...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${HOME}/pillsnap/.venv"

# 기본값 설정
STAGE="${1:-2}"
EXP_DIR="${2:-/home/max16/pillsnap_data/exp/exp01}"
SAVE_REPORT="${3:-false}"

# 색상 출력을 위한 함수
print_header() {
    echo
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    🎯 Stage $STAGE 성능 평가                      ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
}

print_section() {
    echo "║ 📋 $1"
}

print_success() {
    echo "║ ✅ $1"
}

print_error() {
    echo "║ ❌ $1"
}

print_info() {
    echo "║ ℹ️  $1"
}

print_footer() {
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo
}

# 사용법 출력
usage() {
    cat <<EOF
🎯 PillSnap ML Stage Performance Evaluator

사용법:
  $0 [STAGE] [EXP_DIR] [SAVE_REPORT]

매개변수:
  STAGE       평가할 Stage 번호 (1-4, 기본값: 2)
  EXP_DIR     실험 디렉터리 경로 (기본값: /home/max16/pillsnap_data/exp/exp01)
  SAVE_REPORT 상세 리포트 저장 여부 (true/false, 기본값: false)

예시:
  $0 2                                    # Stage 2 기본 평가
  $0 2 /path/to/exp true                  # Stage 2 평가 + 리포트 저장
  $0 3 /home/max16/pillsnap_data/exp/exp02    # Stage 3 평가

지원되는 Stage:
  - Stage 1: 파이프라인 검증 (5K 샘플, 50클래스)
  - Stage 2: 성능 기준선 (25K 샘플, 250클래스) ⭐
  - Stage 3: 확장성 테스트 (100K 샘플, 1K클래스)
  - Stage 4: 최종 프로덕션 (500K 샘플, 5K클래스)
EOF
}

# Stage 검증
validate_stage() {
    if [[ ! "$STAGE" =~ ^[1-4]$ ]]; then
        print_error "잘못된 Stage 번호: $STAGE (1-4만 지원)"
        echo
        usage
        exit 1
    fi
}

# 환경 검증
validate_environment() {
    print_section "환경 검증 중..."
    
    # Virtual Environment 확인
    if [[ ! -d "$VENV_DIR" ]]; then
        print_error "Virtual Environment 없음: $VENV_DIR"
        print_info "먼저 가상환경을 설정하세요: python -m venv $VENV_DIR"
        exit 1
    fi
    
    print_success "Virtual Environment 확인"
    
    # Python 활성화 및 패키지 확인
    source "$VENV_DIR/bin/activate"
    
    # 필수 Python 모듈 확인
    python -c "import torch, torchvision" 2>/dev/null || {
        print_error "PyTorch 패키지 없음"
        print_info "설치 명령: pip install torch torchvision"
        exit 1
    }
    
    print_success "PyTorch 패키지 확인"
    
    # 실험 디렉터리 확인
    if [[ ! -d "$EXP_DIR" ]]; then
        print_error "실험 디렉터리 없음: $EXP_DIR"
        print_info "먼저 학습을 실행하여 실험 디렉터리를 생성하세요"
        exit 1
    fi
    
    print_success "실험 디렉터리 확인: $EXP_DIR"
}

# Stage별 평가기 실행
run_stage_evaluator() {
    print_section "Stage $STAGE 평가 실행 중..."
    
    # 작업 디렉터리 변경
    cd "$ROOT_DIR"
    
    # Virtual Environment 활성화
    source "$VENV_DIR/bin/activate"
    
    # Stage별 평가기 실행
    case $STAGE in
        1)
            print_info "Stage 1 평가기 실행 중..."
            if [[ -f "tests/performance/stage_1_evaluator.py" ]]; then
                python -m tests.performance.stage_1_evaluator \
                    --exp-dir "$EXP_DIR" \
                    ${SAVE_REPORT:+--save-report}
            else
                print_error "Stage 1 평가기 파일 없음"
                exit 1
            fi
            ;;
        2)
            print_info "Stage 2 평가기 실행 중..."
            python -m tests.performance.stage_2_evaluator \
                --exp-dir "$EXP_DIR" \
                ${SAVE_REPORT:+--save-report}
            ;;
        3)
            print_info "Stage 3 평가기 실행 중..."
            if [[ -f "tests/performance/stage_3_evaluator.py" ]]; then
                python -m tests.performance.stage_3_evaluator \
                    --exp-dir "$EXP_DIR" \
                    ${SAVE_REPORT:+--save-report}
            else
                print_error "Stage 3 평가기 파일 없음"
                exit 1
            fi
            ;;
        4)
            print_info "Stage 4 평가기 실행 중..."
            if [[ -f "tests/performance/stage_4_evaluator.py" ]]; then
                python -m tests.performance.stage_4_evaluator \
                    --exp-dir "$EXP_DIR" \
                    ${SAVE_REPORT:+--save-report}
            else
                print_error "Stage 4 평가기 파일 없음"
                exit 1
            fi
            ;;
        *)
            print_error "알 수 없는 Stage: $STAGE"
            exit 1
            ;;
    esac
}

# 평가 후 요약 출력
print_evaluation_summary() {
    print_section "평가 완료 요약"
    
    # 평가 리포트 파일 확인
    REPORT_FILE="$EXP_DIR/reports/stage_${STAGE}_evaluation.json"
    
    if [[ -f "$REPORT_FILE" ]]; then
        print_success "평가 리포트 생성됨: $REPORT_FILE"
        
        # JSON 파일에서 주요 정보 추출 (jq 사용 가능한 경우)
        if command -v jq >/dev/null 2>&1; then
            print_info "주요 결과:"
            
            # 성능 점수
            PERFORMANCE_SCORE=$(jq -r '.recommendation.performance_score // "N/A"' "$REPORT_FILE" 2>/dev/null || echo "N/A")
            echo "║   📊 성능 점수: $PERFORMANCE_SCORE"
            
            # 권장사항
            DECISION=$(jq -r '.recommendation.decision // "N/A"' "$REPORT_FILE" 2>/dev/null || echo "N/A")
            echo "║   🎯 권장사항: $DECISION"
            
            # 분류 정확도
            ACCURACY=$(jq -r '.performance_metrics.classification_accuracy // "N/A"' "$REPORT_FILE" 2>/dev/null || echo "N/A")
            echo "║   🎯 분류 정확도: $ACCURACY"
        fi
    else
        print_info "평가 리포트 파일이 생성되지 않았습니다"
    fi
    
    # 로그 디렉터리 안내
    if [[ -d "$EXP_DIR/logs" ]]; then
        print_info "학습 로그: $EXP_DIR/logs/"
    fi
    
    # TensorBoard 안내
    if [[ -d "$EXP_DIR/tb" ]]; then
        print_info "TensorBoard 로그: $EXP_DIR/tb/"
        print_info "TensorBoard 실행: tensorboard --logdir=$EXP_DIR/tb"
    fi
}

# 다음 단계 안내
print_next_steps() {
    print_section "다음 단계 안내"
    
    case $STAGE in
        1)
            print_info "Stage 1 완료 후 → Stage 2 진행"
            print_info "명령어: python -m src.training.train_classification_stage --stage 2"
            ;;
        2)
            print_info "Stage 2 완료 후 → Stage 3 진행"
            print_info "명령어: python -m src.training.train_classification_stage --stage 3"
            ;;
        3)
            print_info "Stage 3 완료 후 → Stage 4 진행 (최종 프로덕션)"
            print_info "명령어: python -m src.training.train_classification_stage --stage 4"
            ;;
        4)
            print_info "Stage 4 완료! 🎉"
            print_info "프로덕션 배포 준비: bash scripts/deployment/run_api.sh"
            ;;
    esac
}

# 오류 처리
handle_error() {
    local exit_code=$?
    print_error "Stage $STAGE 평가 실패 (종료 코드: $exit_code)"
    
    # 로그 파일 확인 안내
    if [[ -f "$EXP_DIR/logs/train.err" ]]; then
        print_info "오류 로그 확인: $EXP_DIR/logs/train.err"
    fi
    
    print_info "디버깅을 위한 수동 실행:"
    print_info "cd $ROOT_DIR"
    print_info "source $VENV_DIR/bin/activate"
    print_info "python -m tests.performance.stage_${STAGE}_evaluator --exp-dir $EXP_DIR"
    
    print_footer
    exit $exit_code
}

# 시그널 핸들러
cleanup() {
    print_info "평가 중단됨 (Ctrl+C)"
    print_footer
    exit 130
}

# 메인 실행 함수
main() {
    # Help 옵션 처리
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi
    
    # 시그널 핸들러 등록
    trap cleanup SIGINT SIGTERM
    trap handle_error ERR
    
    # 헤더 출력
    print_header
    
    # 환경 변수 출력
    print_info "Stage: $STAGE"
    print_info "실험 디렉터리: $EXP_DIR"
    print_info "리포트 저장: $SAVE_REPORT"
    echo "║"
    
    # 단계별 실행
    validate_stage
    validate_environment
    run_stage_evaluator
    echo "║"
    print_evaluation_summary
    echo "║"
    print_next_steps
    
    # 푸터 출력
    print_footer
    
    print_success "Stage $STAGE 평가 완료!"
}

# 스크립트 시작점
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi