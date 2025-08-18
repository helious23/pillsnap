#!/bin/bash
# 릴리스 아카이브 생성 스크립트
# 배포 산출물 표준화 및 무결성 검증

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
CONFIG_FILE="$PROJECT_ROOT/config.yaml"

# 기본값 설정
DEFAULT_EXP_DIR="/mnt/data/exp/exp01"
INCLUDE_ALL_ONNX=false
ONNX_PATH=""

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
PillSnap ML 릴리스 아카이브 생성 도구

사용법:
    $0 [옵션]

옵션:
    -h, --help              이 도움말 표시
    -o, --onnx PATH         특정 ONNX 파일 경로 지정
    -a, --all-onnx          모든 ONNX 파일 포함 (기본: 최신만)
    -e, --exp-dir DIR       실험 디렉토리 경로 (기본: $DEFAULT_EXP_DIR)

예시:
    $0                                          # 기본 설정으로 아카이브 생성
    $0 -o /mnt/data/exp/exp01/export/model.onnx # 특정 ONNX 포함
    $0 -a                                       # 모든 ONNX 파일 포함
    $0 -e /mnt/data/exp/exp02                   # 다른 실험 디렉토리

생성 파일:
    - release-YYYYMMDD-HHMMSS-{sha|nogit}.tar.gz
    - release-YYYYMMDD-HHMMSS-{sha|nogit}.tar.gz.sha256
    - MANIFEST.json (아카이브 내 파일 목록)
    - VERSION.txt (버전 정보)
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
            -o|--onnx)
                ONNX_PATH="$2"
                shift 2
                ;;
            -a|--all-onnx)
                INCLUDE_ALL_ONNX=true
                shift
                ;;
            -e|--exp-dir)
                DEFAULT_EXP_DIR="$2"
                shift 2
                ;;
            *)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Git SHA 획득
get_git_sha() {
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git rev-parse --short HEAD
    else
        echo "nogit"
    fi
}

# 실험 디렉토리에서 설정 로드
load_exp_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        # yq가 있으면 사용, 없으면 기본값
        if command -v yq &> /dev/null; then
            EXP_DIR=$(yq '.paths.exp_dir' "$CONFIG_FILE" 2>/dev/null || echo "$DEFAULT_EXP_DIR")
            if [[ "$EXP_DIR" == "null" || -z "$EXP_DIR" ]]; then
                EXP_DIR="$DEFAULT_EXP_DIR"
            fi
        else
            EXP_DIR="$DEFAULT_EXP_DIR"
            log_warning "yq not found, using default exp_dir: $EXP_DIR"
        fi
    else
        EXP_DIR="$DEFAULT_EXP_DIR"
        log_warning "config.yaml not found, using default exp_dir: $EXP_DIR"
    fi
    
    log_info "Experiment directory: $EXP_DIR"
}

# 포함할 ONNX 파일들 결정
determine_onnx_files() {
    local export_dir="$EXP_DIR/export"
    local onnx_files=()
    
    if [[ -n "$ONNX_PATH" ]]; then
        # 특정 ONNX 파일 지정
        if [[ -f "$ONNX_PATH" ]]; then
            onnx_files=("$ONNX_PATH")
            log_info "Using specified ONNX: $ONNX_PATH"
        else
            log_error "Specified ONNX file not found: $ONNX_PATH"
            exit 1
        fi
    elif [[ "$INCLUDE_ALL_ONNX" == "true" ]]; then
        # 모든 ONNX 파일 포함
        if [[ -d "$export_dir" ]]; then
            mapfile -t onnx_files < <(find "$export_dir" -name "*.onnx" -type f)
            log_info "Including all ONNX files: ${#onnx_files[@]} found"
        fi
    else
        # 최신 ONNX 파일들 (심볼릭 링크 우선)
        local latest_files=()
        for pattern in "latest_*.onnx" "*.onnx"; do
            if [[ -d "$export_dir" ]]; then
                while IFS= read -r -d '' file; do
                    latest_files+=("$file")
                done < <(find "$export_dir" -name "$pattern" -type f -print0 | head -z -n 2)
            fi
        done
        
        if [[ ${#latest_files[@]} -gt 0 ]]; then
            # 중복 제거 및 정렬
            mapfile -t onnx_files < <(printf '%s\n' "${latest_files[@]}" | sort -u)
            log_info "Using latest ONNX files: ${#onnx_files[@]} found"
        fi
    fi
    
    if [[ ${#onnx_files[@]} -eq 0 ]]; then
        log_warning "No ONNX files found in $export_dir"
    fi
    
    printf '%s\n' "${onnx_files[@]}"
}

# VERSION.txt 생성
create_version_file() {
    local temp_dir="$1"
    local git_sha="$2"
    local version_file="$temp_dir/VERSION.txt"
    
    cat > "$version_file" << EOF
PillSnap ML Release Information
===============================

Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Git SHA: $git_sha
PyTorch Version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
ONNX Version: $(python -c "import onnx; print(onnx.__version__)" 2>/dev/null || echo "unknown")
ONNX Runtime: $(python -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")

Build Environment:
- OS: $(uname -a)
- Python: $(python --version 2>&1)
- CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")

Configuration:
- Experiment Directory: $EXP_DIR
- Include All ONNX: $INCLUDE_ALL_ONNX
- Specified ONNX: ${ONNX_PATH:-"auto"}
EOF
    
    log_info "Created VERSION.txt"
}

# MANIFEST.json 생성
create_manifest() {
    local temp_dir="$1"
    local manifest_file="$temp_dir/MANIFEST.json"
    
    log_info "Generating file manifest..."
    
    # JSON 형식으로 파일 목록 생성
    echo '{"files": [' > "$manifest_file"
    
    local first=true
    while IFS= read -r -d '' file; do
        local rel_path="${file#$temp_dir/}"
        local size=$(stat -c %s "$file")
        local sha256=$(sha256sum "$file" | cut -d' ' -f1)
        local modified=$(stat -c %Y "$file")
        
        if [[ "$first" != "true" ]]; then
            echo "," >> "$manifest_file"
        fi
        first=false
        
        cat >> "$manifest_file" << EOF
  {
    "path": "$rel_path",
    "size": $size,
    "sha256": "$sha256",
    "modified": $modified
  }EOF
    done < <(find "$temp_dir" -type f ! -name "MANIFEST.json" -print0)
    
    echo "" >> "$manifest_file"
    echo '], "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}' >> "$manifest_file"
    
    log_info "Created MANIFEST.json"
}

# 아카이브할 파일들 준비
prepare_archive_files() {
    local temp_dir="$1"
    local git_sha="$2"
    
    log_info "Preparing release files..."
    
    # 기본 설정 파일들
    local base_files=(
        "config.yaml"
        ".env.example"
        "requirements.txt"
        "README.md"
    )
    
    for file in "${base_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            cp "$PROJECT_ROOT/$file" "$temp_dir/"
            log_info "Added: $file"
        else
            log_warning "Missing: $file"
        fi
    done
    
    # ONNX 파일들
    local onnx_files
    mapfile -t onnx_files < <(determine_onnx_files)
    
    if [[ ${#onnx_files[@]} -gt 0 ]]; then
        mkdir -p "$temp_dir/export"
        for onnx_file in "${onnx_files[@]}"; do
            if [[ -f "$onnx_file" ]]; then
                cp "$onnx_file" "$temp_dir/export/"
                log_info "Added ONNX: $(basename "$onnx_file")"
            fi
        done
    fi
    
    # Export 리포트
    local export_report="$EXP_DIR/export/export_report.json"
    if [[ -f "$export_report" ]]; then
        cp "$export_report" "$temp_dir/export/"
        log_info "Added: export_report.json"
    fi
    
    # 최신 메트릭스 리포트
    local metrics_dir="$EXP_DIR/reports"
    if [[ -d "$metrics_dir" ]]; then
        mkdir -p "$temp_dir/reports"
        
        # 최신 metrics.json 찾기
        local latest_metrics
        latest_metrics=$(find "$metrics_dir" -name "metrics*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || echo "")
        
        if [[ -n "$latest_metrics" && -f "$latest_metrics" ]]; then
            cp "$latest_metrics" "$temp_dir/reports/metrics.json"
            log_info "Added: latest metrics report"
        fi
    fi
    
    # 버전 정보 생성
    create_version_file "$temp_dir" "$git_sha"
    
    # 매니페스트 생성
    create_manifest "$temp_dir"
}

# 아카이브 생성 및 체크섬
create_archive() {
    local temp_dir="$1"
    local timestamp="$2"
    local git_sha="$3"
    
    local releases_dir="$EXP_DIR/releases"
    mkdir -p "$releases_dir"
    
    local archive_name="release-$timestamp-$git_sha.tar.gz"
    local archive_path="$releases_dir/$archive_name"
    
    log_info "Creating archive: $archive_name"
    
    # tar.gz 생성
    tar -czf "$archive_path" -C "$temp_dir" .
    
    # SHA256 체크섬 생성
    local checksum_file="$archive_path.sha256"
    (cd "$releases_dir" && sha256sum "$archive_name" > "$(basename "$checksum_file")")
    
    # 결과 출력
    local archive_size_mb=$(($(stat -c %s "$archive_path") / 1024 / 1024))
    
    log_success "Archive created successfully!"
    echo ""
    echo "📦 Release Archive:"
    echo "   File: $archive_path"
    echo "   Size: ${archive_size_mb}MB"
    echo "   SHA256: $checksum_file"
    echo ""
    echo "Contents:"
    tar -tzf "$archive_path" | head -20
    if [[ $(tar -tzf "$archive_path" | wc -l) -gt 20 ]]; then
        echo "   ... and $(($(tar -tzf "$archive_path" | wc -l) - 20)) more files"
    fi
    echo ""
    
    # 복구 안내
    echo "📋 Recovery Instructions:"
    echo "   1. Extract: tar -xzf $archive_name"
    echo "   2. Verify: sha256sum -c $(basename "$checksum_file")"
    echo "   3. Load model: Update config paths to extracted ONNX files"
    echo "   4. API reload: POST /reload with new model paths"
    echo ""
}

# 정리 작업
cleanup() {
    if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# 메인 실행
main() {
    log_info "PillSnap ML Release Archive Creator"
    echo "======================================"
    
    # 명령행 인자 파싱
    parse_arguments "$@"
    
    # 실험 설정 로드
    load_exp_config
    
    # 실험 디렉토리 존재 확인
    if [[ ! -d "$EXP_DIR" ]]; then
        log_error "Experiment directory not found: $EXP_DIR"
        exit 1
    fi
    
    # 임시 디렉토리 생성
    TEMP_DIR=$(mktemp -d)
    trap cleanup EXIT
    
    # 타임스탬프 및 Git SHA
    local timestamp=$(date -u '+%Y%m%d-%H%M%S')
    local git_sha=$(get_git_sha)
    
    log_info "Timestamp: $timestamp"
    log_info "Git SHA: $git_sha"
    echo ""
    
    # 아카이브 파일들 준비
    prepare_archive_files "$TEMP_DIR" "$git_sha"
    echo ""
    
    # 아카이브 생성
    create_archive "$TEMP_DIR" "$timestamp" "$git_sha"
    
    log_success "Release archive creation completed!"
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi