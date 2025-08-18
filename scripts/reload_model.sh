#!/bin/bash
# 무중단 모델 교체 스크립트
# /reload API를 통한 안전한 모델 업데이트

set -euo pipefail

# 색깔 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 설정값들
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 기본 설정
API_URL="http://localhost:8000"
API_KEY=""
MODEL_PATH=""
MODEL_TYPE=""
TIMEOUT=30
VERIFY_AFTER=true

# 로그 함수들
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

# 도움말 출력
show_help() {
    cat << EOF
PillSnap ML 무중단 모델 교체 도구

사용법:
    $0 --path MODEL_PATH [옵션]

필수 옵션:
    -p, --path PATH         교체할 모델 파일 경로 (.pt 또는 .onnx)

선택 옵션:
    -h, --help              이 도움말 표시
    -u, --url URL           API 서버 URL (기본: $API_URL)
    -k, --key KEY           API 키 (기본: .env에서 로드)
    -t, --timeout SEC       API 타임아웃 (기본: ${TIMEOUT}초)
    --no-verify             교체 후 검증 생략
    --detection             검출 모델 교체
    --classification        분류 모델 교체 (기본: 자동감지)

예시:
    $0 -p /mnt/data/exp/exp01/export/classification-20250817-abc1234.onnx
    $0 -p /mnt/data/exp/exp01/checkpoints/best.pt -k YOUR_API_KEY
    $0 -p detection_model.onnx --detection -u http://api.pillsnap.co.kr
    $0 -p classification_model.pt --classification --no-verify

환경변수:
    PILLSNAP_API_KEY        API 키 (--key 옵션보다 우선순위 낮음)
    PILLSNAP_API_URL        API URL (--url 옵션보다 우선순위 낮음)
EOF
}

# 명령행 인자 파싱
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -p|--path)
                MODEL_PATH="$2"
                shift 2
                ;;
            -u|--url)
                API_URL="$2"
                shift 2
                ;;
            -k|--key)
                API_KEY="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --no-verify)
                VERIFY_AFTER=false
                shift
                ;;
            --detection)
                MODEL_TYPE="detection"
                shift
                ;;
            --classification)
                MODEL_TYPE="classification"
                shift
                ;;
            *)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 필수 인자 검증
    if [[ -z "$MODEL_PATH" ]]; then
        log_error "모델 경로를 지정해야 합니다 (-p/--path)"
        show_help
        exit 1
    fi
}

# API 키 로드
load_api_key() {
    # 명령행에서 지정되지 않은 경우 환경변수나 .env에서 로드
    if [[ -z "$API_KEY" ]]; then
        if [[ -n "${PILLSNAP_API_KEY:-}" ]]; then
            API_KEY="$PILLSNAP_API_KEY"
            log_info "API 키를 환경변수에서 로드"
        elif [[ -f "$PROJECT_ROOT/.env" ]]; then
            # .env 파일에서 API_KEY 추출
            if grep -q "^API_KEY=" "$PROJECT_ROOT/.env"; then
                API_KEY=$(grep "^API_KEY=" "$PROJECT_ROOT/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
                log_info "API 키를 .env 파일에서 로드"
            fi
        fi
    fi
    
    if [[ -z "$API_KEY" ]]; then
        log_error "API 키가 필요합니다. -k 옵션, PILLSNAP_API_KEY 환경변수, 또는 .env 파일에서 API_KEY를 설정하세요."
        exit 1
    fi
}

# API URL 설정
setup_api_url() {
    # 환경변수에서 URL 로드 (명령행 옵션이 우선)
    if [[ "$API_URL" == "http://localhost:8000" && -n "${PILLSNAP_API_URL:-}" ]]; then
        API_URL="$PILLSNAP_API_URL"
        log_info "API URL을 환경변수에서 로드: $API_URL"
    fi
    
    # URL 정규화
    API_URL="${API_URL%/}"  # 끝의 슬래시 제거
    
    log_info "API URL: $API_URL"
}

# 모델 파일 검증
validate_model_file() {
    if [[ ! -f "$MODEL_PATH" ]]; then
        log_error "모델 파일을 찾을 수 없습니다: $MODEL_PATH"
        exit 1
    fi
    
    # 절대 경로로 변환
    MODEL_PATH=$(realpath "$MODEL_PATH")
    
    # 파일 확장자로 모델 타입 자동 감지
    if [[ -z "$MODEL_TYPE" ]]; then
        local filename=$(basename "$MODEL_PATH")
        if [[ "$filename" == *"detection"* ]]; then
            MODEL_TYPE="detection"
        elif [[ "$filename" == *"classification"* ]]; then
            MODEL_TYPE="classification"
        elif [[ "$filename" == *".onnx" || "$filename" == *".pt" ]]; then
            # 기본값: classification
            MODEL_TYPE="classification"
            log_warning "모델 타입을 자동감지했습니다: $MODEL_TYPE"
        else
            log_error "모델 타입을 감지할 수 없습니다. --detection 또는 --classification 옵션을 사용하세요."
            exit 1
        fi
    fi
    
    local file_size=$(stat -c %s "$MODEL_PATH")
    local file_size_mb=$((file_size / 1024 / 1024))
    
    log_info "모델 파일: $MODEL_PATH"
    log_info "모델 타입: $MODEL_TYPE"
    log_info "파일 크기: ${file_size_mb}MB"
}

# API 서버 상태 확인
check_api_status() {
    log_info "API 서버 상태 확인 중..."
    
    local health_url="$API_URL/health"
    
    if ! curl -s --max-time 10 "$health_url" > /dev/null; then
        log_error "API 서버에 연결할 수 없습니다: $health_url"
        log_error "다음을 확인하세요:"
        log_error "  1. API 서버가 실행 중인지 확인"
        log_error "  2. URL이 올바른지 확인: $API_URL"
        log_error "  3. 네트워크 연결 상태 확인"
        exit 1
    fi
    
    log_success "API 서버 연결 확인 ✓"
}

# 현재 모델 버전 확인
get_current_version() {
    log_info "현재 모델 버전 확인 중..."
    
    local version_url="$API_URL/version"
    local response
    
    response=$(curl -s --max-time "$TIMEOUT" \
        -H "X-API-Key: $API_KEY" \
        -H "Accept: application/json" \
        "$version_url") || {
        log_error "버전 정보를 가져올 수 없습니다"
        return 1
    }
    
    # JSON 파싱 (jq가 있으면 사용, 없으면 grep)
    if command -v jq &> /dev/null; then
        local current_model_path=$(echo "$response" | jq -r ".${MODEL_TYPE}_model_path // \"unknown\"")
        local current_version=$(echo "$response" | jq -r '.version_tag // "unknown"')
        
        log_info "현재 ${MODEL_TYPE} 모델: $current_model_path"
        log_info "현재 버전: $current_version"
    else
        log_info "현재 버전 정보:"
        echo "$response" | grep -E "(${MODEL_TYPE}_model_path|version_tag)" || echo "$response"
    fi
    
    echo "$response"
}

# 모델 교체 실행
reload_model() {
    log_info "모델 교체 실행 중..."
    
    local reload_url="$API_URL/reload"
    local json_payload
    
    # JSON 페이로드 구성
    if [[ "$MODEL_TYPE" == "detection" ]]; then
        json_payload="{\"detection_path\": \"$MODEL_PATH\"}"
    else
        json_payload="{\"classification_path\": \"$MODEL_PATH\"}"
    fi
    
    log_info "요청 페이로드: $json_payload"
    
    # API 호출
    local response
    local http_code
    
    response=$(curl -s --max-time "$TIMEOUT" \
        -w "%{http_code}" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "$json_payload" \
        "$reload_url") || {
        log_error "모델 교체 요청 실패"
        return 1
    }
    
    # HTTP 상태 코드와 응답 본문 분리
    http_code="${response: -3}"
    response_body="${response%???}"
    
    log_info "HTTP 상태 코드: $http_code"
    
    if [[ "$http_code" == "200" ]]; then
        log_success "모델 교체 성공!"
        
        # 응답 파싱
        if command -v jq &> /dev/null; then
            echo "$response_body" | jq '.'
        else
            echo "$response_body"
        fi
        
        return 0
    else
        log_error "모델 교체 실패 (HTTP $http_code)"
        log_error "응답: $response_body"
        
        # 일반적인 오류 원인 안내
        case "$http_code" in
            401|403)
                log_error "인증 실패: API 키를 확인하세요"
                ;;
            404)
                log_error "API 엔드포인트를 찾을 수 없습니다: $reload_url"
                ;;
            422)
                log_error "요청 형식 오류: 모델 경로나 파일 형식을 확인하세요"
                ;;
            500)
                log_error "서버 오류: 모델 파일 손상이나 호환성 문제일 수 있습니다"
                ;;
            *)
                log_error "예상치 못한 오류"
                ;;
        esac
        
        return 1
    fi
}

# 교체 후 검증
verify_reload() {
    if [[ "$VERIFY_AFTER" == "false" ]]; then
        log_info "검증 생략됨 (--no-verify)"
        return 0
    fi
    
    log_info "모델 교체 검증 중..."
    
    # 잠시 대기 (모델 로딩 시간 고려)
    sleep 3
    
    # 새 버전 정보 확인
    local new_version_response
    new_version_response=$(get_current_version) || {
        log_warning "검증을 위한 버전 정보 가져오기 실패"
        return 1
    }
    
    # 헬스체크로 모델 로딩 상태 확인
    local health_url="$API_URL/health"
    local health_response
    
    health_response=$(curl -s --max-time 10 \
        -H "Accept: application/json" \
        "$health_url") || {
        log_warning "헬스체크 실패"
        return 1
    }
    
    # 모델 로딩 상태 확인
    if command -v jq &> /dev/null; then
        local models_loaded=$(echo "$health_response" | jq -r '.models_loaded // {}')
        local classification_loaded=$(echo "$models_loaded" | jq -r '.classification // false')
        local detection_loaded=$(echo "$models_loaded" | jq -r '.detection // false')
        
        if [[ "$MODEL_TYPE" == "classification" && "$classification_loaded" == "true" ]]; then
            log_success "분류 모델 로딩 확인 ✓"
        elif [[ "$MODEL_TYPE" == "detection" && "$detection_loaded" == "true" ]]; then
            log_success "검출 모델 로딩 확인 ✓"
        else
            log_warning "모델 로딩 상태 불확실: $health_response"
            return 1
        fi
    else
        log_info "헬스체크 응답: $health_response"
    fi
    
    log_success "모델 교체 검증 완료 ✓"
    return 0
}

# 메인 실행
main() {
    log_info "PillSnap ML 무중단 모델 교체"
    echo "======================================"
    
    # 인자 파싱 및 검증
    parse_arguments "$@"
    load_api_key
    setup_api_url
    validate_model_file
    
    echo ""
    log_info "모델 교체 준비 완료"
    log_info "  모델: $MODEL_PATH"
    log_info "  타입: $MODEL_TYPE"
    log_info "  API: $API_URL"
    echo ""
    
    # 사전 검사
    check_api_status
    echo ""
    
    # 현재 상태 확인
    log_info "=== 교체 전 상태 ==="
    get_current_version > /dev/null
    echo ""
    
    # 모델 교체 실행
    log_info "=== 모델 교체 실행 ==="
    if reload_model; then
        echo ""
        
        # 검증
        log_info "=== 교체 후 검증 ==="
        if verify_reload; then
            echo ""
            log_success "======================================"
            log_success "모델 교체가 성공적으로 완료되었습니다!"
            log_success "======================================"
        else
            echo ""
            log_warning "모델 교체는 완료되었지만 검증에서 문제가 발견되었습니다."
            log_warning "API 서버와 모델 상태를 수동으로 확인하세요."
        fi
    else
        echo ""
        log_error "======================================"
        log_error "모델 교체에 실패했습니다."
        log_error "======================================"
        echo ""
        log_info "문제 해결 방법:"
        log_info "  1. 모델 파일 경로와 권한 확인"
        log_info "  2. API 키 확인"
        log_info "  3. 모델 파일 형식과 호환성 확인"
        log_info "  4. API 서버 로그 확인"
        exit 1
    fi
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi