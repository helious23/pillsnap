#!/bin/bash
# PillSnap ML 시스템 유지보수 스크립트
# 로그 정리, 디스크 관리, 아티팩트 정리

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

# 기본 설정
DEFAULT_EXP_DIR="/mnt/data/exp/exp01"
DRY_RUN=false
VERBOSE=false

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

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${NC}[VERBOSE]${NC} $1"
    fi
}

# 도움말 출력
show_help() {
    cat << EOF
PillSnap ML 시스템 유지보수 도구

사용법:
    $0 [옵션]

옵션:
    -h, --help              이 도움말 표시
    -n, --dry-run           실제 삭제 없이 시뮬레이션만 실행
    -v, --verbose           상세 출력 모드
    -e, --exp-dir DIR       실험 디렉토리 경로 (기본: auto-detect)

작업 내용:
    1. 로그 파일 정리 및 압축 (7일 이상)
    2. 체크포인트 정리 (14일 이상, best.pt 보존)
    3. ONNX 파일 정리 (5개 초과 시 오래된 것 제거)
    4. 임시 파일 정리
    5. 디스크 사용량 리포트 생성

예시:
    $0                      # 기본 유지보수 실행
    $0 -n                   # 드라이런 모드 (실제 삭제 없음)
    $0 -v                   # 상세 출력 모드
    $0 -e /mnt/data/exp/exp02  # 특정 실험 디렉토리
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
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
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

# 실험 디렉토리 로드
load_exp_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        if command -v yq &> /dev/null; then
            EXP_DIR=$(yq '.paths.exp_dir' "$CONFIG_FILE" 2>/dev/null || echo "$DEFAULT_EXP_DIR")
            if [[ "$EXP_DIR" == "null" || -z "$EXP_DIR" ]]; then
                EXP_DIR="$DEFAULT_EXP_DIR"
            fi
        else
            EXP_DIR="$DEFAULT_EXP_DIR"
        fi
    else
        EXP_DIR="$DEFAULT_EXP_DIR"
    fi
    
    log_info "Experiment directory: $EXP_DIR"
}

# 파일 크기를 사람이 읽기 쉬운 형태로 변환
human_readable_size() {
    local size=$1
    if (( size >= 1073741824 )); then
        printf "%.1fGB" $(( size / 1073741824 ))
    elif (( size >= 1048576 )); then
        printf "%.1fMB" $(( size / 1048576 ))
    elif (( size >= 1024 )); then
        printf "%.1fKB" $(( size / 1024 ))
    else
        printf "%dB" $size
    fi
}

# 로그 파일 정리
cleanup_logs() {
    log_info "로그 파일 정리 중..."
    
    local logs_dir="$EXP_DIR/logs"
    local archive_dir="$logs_dir/archive"
    
    if [[ ! -d "$logs_dir" ]]; then
        log_warning "로그 디렉토리 없음: $logs_dir"
        return
    fi
    
    # 아카이브 디렉토리 생성
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$archive_dir"
    fi
    
    local files_compressed=0
    local files_deleted=0
    local total_saved=0
    
    # 7일 이상된 .out, .err 파일들을 gzip으로 압축
    while IFS= read -r -d '' file; do
        local filename=$(basename "$file")
        local compressed_file="$archive_dir/${filename}.gz"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_verbose "Would compress: $file"
        else
            gzip -c "$file" > "$compressed_file"
            local original_size=$(stat -c %s "$file")
            local compressed_size=$(stat -c %s "$compressed_file")
            total_saved=$((total_saved + original_size - compressed_size))
            
            rm "$file"
            log_verbose "Compressed: $filename (saved $(human_readable_size $((original_size - compressed_size))))"
        fi
        
        files_compressed=$((files_compressed + 1))
    done < <(find "$logs_dir" -maxdepth 1 -name "*.out" -o -name "*.err" -type f -mtime +7 -print0)
    
    # 30일 이상된 압축 파일들 삭제
    while IFS= read -r -d '' file; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log_verbose "Would delete old archive: $file"
        else
            rm "$file"
            log_verbose "Deleted old archive: $(basename "$file")"
        fi
        
        files_deleted=$((files_deleted + 1))
    done < <(find "$archive_dir" -name "*.gz" -type f -mtime +30 -print0 2>/dev/null)
    
    if [[ $files_compressed -gt 0 || $files_deleted -gt 0 ]]; then
        log_success "로그 정리 완료: ${files_compressed}개 압축, ${files_deleted}개 삭제"
        if [[ $total_saved -gt 0 ]]; then
            log_info "디스크 절약: $(human_readable_size $total_saved)"
        fi
    else
        log_info "정리할 로그 파일 없음"
    fi
}

# 체크포인트 정리
cleanup_checkpoints() {
    log_info "체크포인트 정리 중..."
    
    local ckpt_dir="$EXP_DIR/checkpoints"
    
    if [[ ! -d "$ckpt_dir" ]]; then
        log_warning "체크포인트 디렉토리 없음: $ckpt_dir"
        return
    fi
    
    local files_deleted=0
    local total_freed=0
    
    # best.pt는 보존, 14일 이상된 last_*.pt 파일들 삭제
    while IFS= read -r -d '' file; do
        local filename=$(basename "$file")
        
        # best.pt는 건드리지 않음
        if [[ "$filename" == "best.pt" || "$filename" == *"best"* ]]; then
            log_verbose "Preserving: $filename"
            continue
        fi
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_verbose "Would delete checkpoint: $file"
        else
            local file_size=$(stat -c %s "$file")
            rm "$file"
            total_freed=$((total_freed + file_size))
            log_verbose "Deleted checkpoint: $filename ($(human_readable_size $file_size))"
        fi
        
        files_deleted=$((files_deleted + 1))
    done < <(find "$ckpt_dir" -name "last_*.pt" -o -name "epoch_*.pt" -type f -mtime +14 -print0)
    
    if [[ $files_deleted -gt 0 ]]; then
        log_success "체크포인트 정리 완료: ${files_deleted}개 삭제"
        if [[ $total_freed -gt 0 ]]; then
            log_info "디스크 절약: $(human_readable_size $total_freed)"
        fi
    else
        log_info "정리할 체크포인트 없음"
    fi
}

# ONNX 파일 정리
cleanup_onnx_files() {
    log_info "ONNX 파일 정리 중..."
    
    local export_dir="$EXP_DIR/export"
    
    if [[ ! -d "$export_dir" ]]; then
        log_warning "Export 디렉토리 없음: $export_dir"
        return
    fi
    
    # ONNX 파일들을 수정 시간순으로 정렬
    local onnx_files=()
    while IFS= read -r -d '' file; do
        # 심볼릭 링크는 제외
        if [[ ! -L "$file" ]]; then
            onnx_files+=("$file")
        fi
    done < <(find "$export_dir" -name "*.onnx" -type f -printf '%T@ %p\0' | sort -z -n | cut -z -d' ' -f2-)
    
    local total_files=${#onnx_files[@]}
    local files_to_keep=5
    local files_deleted=0
    local total_freed=0
    
    if [[ $total_files -gt $files_to_keep ]]; then
        local files_to_delete=$((total_files - files_to_keep))
        
        log_info "ONNX 파일 ${total_files}개 중 최신 ${files_to_keep}개 유지, ${files_to_delete}개 삭제 예정"
        
        # 오래된 파일들 삭제 (처음 N개가 가장 오래된 것들)
        for ((i=0; i<files_to_delete; i++)); do
            local file="${onnx_files[i]}"
            local filename=$(basename "$file")
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_verbose "Would delete ONNX: $filename"
            else
                local file_size=$(stat -c %s "$file")
                rm "$file"
                total_freed=$((total_freed + file_size))
                log_verbose "Deleted ONNX: $filename ($(human_readable_size $file_size))"
            fi
            
            files_deleted=$((files_deleted + 1))
        done
        
        log_success "ONNX 정리 완료: ${files_deleted}개 삭제"
        if [[ $total_freed -gt 0 ]]; then
            log_info "디스크 절약: $(human_readable_size $total_freed)"
        fi
    else
        log_info "ONNX 파일 ${total_files}개, 정리 불필요"
    fi
}

# 임시 파일 정리
cleanup_temp_files() {
    log_info "임시 파일 정리 중..."
    
    local temp_patterns=(
        "$EXP_DIR/tmp"
        "/tmp/shard_*"
        "/tmp/pillsnap_*"
        "/tmp/torch_*"
        "/tmp/onnx_*"
    )
    
    local files_deleted=0
    local total_freed=0
    
    for pattern in "${temp_patterns[@]}"; do
        if [[ -d "$pattern" ]] || ls $pattern 1> /dev/null 2>&1; then
            while IFS= read -r -d '' file; do
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_verbose "Would delete temp: $file"
                else
                    if [[ -d "$file" ]]; then
                        local dir_size=$(du -sb "$file" 2>/dev/null | cut -f1 || echo 0)
                        rm -rf "$file"
                        total_freed=$((total_freed + dir_size))
                        log_verbose "Deleted temp dir: $file ($(human_readable_size $dir_size))"
                    else
                        local file_size=$(stat -c %s "$file" 2>/dev/null || echo 0)
                        rm -f "$file"
                        total_freed=$((total_freed + file_size))
                        log_verbose "Deleted temp file: $file ($(human_readable_size $file_size))"
                    fi
                fi
                
                files_deleted=$((files_deleted + 1))
            done < <(find $pattern -maxdepth 1 -mtime +1 -print0 2>/dev/null)
        fi
    done
    
    if [[ $files_deleted -gt 0 ]]; then
        log_success "임시 파일 정리 완료: ${files_deleted}개 삭제"
        if [[ $total_freed -gt 0 ]]; then
            log_info "디스크 절약: $(human_readable_size $total_freed)"
        fi
    else
        log_info "정리할 임시 파일 없음"
    fi
}

# 디스크 사용량 리포트 생성
generate_disk_report() {
    log_info "디스크 사용량 리포트 생성 중..."
    
    local reports_dir="$EXP_DIR/reports"
    local report_file="$reports_dir/disk_$(date +%Y%m%d).txt"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$reports_dir"
    fi
    
    local report_content="PillSnap ML 디스크 사용량 리포트
==========================================
생성일시: $(date)
실험 디렉토리: $EXP_DIR

전체 시스템 디스크 사용량:
$(df -h / /mnt/data 2>/dev/null || df -h /)

실험 디렉토리별 사용량:
$(if [[ -d "$EXP_DIR" ]]; then du -sh "$EXP_DIR"/* 2>/dev/null | sort -hr || echo "디렉토리 접근 실패"; else echo "실험 디렉토리 없음"; fi)

상위 10개 대용량 파일:
$(find "$EXP_DIR" -type f -exec du -h {} + 2>/dev/null | sort -hr | head -10 || echo "파일 검색 실패")

메모리 사용량:
$(free -h)

GPU 메모리 (가능한 경우):
$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{print "GPU Memory: " $1 "MB / " $2 "MB"}' || echo "NVIDIA GPU 없음")
"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "드라이런 모드: 리포트 미생성"
        echo ""
        echo "=== 디스크 리포트 미리보기 ==="
        echo "$report_content"
        echo "=== 미리보기 종료 ==="
    else
        echo "$report_content" > "$report_file"
        log_success "디스크 리포트 생성: $report_file"
    fi
}

# 메인 실행
main() {
    local start_time=$(date +%s)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "드라이런 모드: 실제 파일 삭제 없음"
    fi
    
    log_info "PillSnap ML 시스템 유지보수 시작"
    echo "======================================"
    
    # 설정 로드
    parse_arguments "$@"
    load_exp_config
    
    # 실험 디렉토리 존재 확인
    if [[ ! -d "$EXP_DIR" ]]; then
        log_error "실험 디렉토리 없음: $EXP_DIR"
        exit 1
    fi
    
    echo ""
    
    # 유지보수 작업 실행
    cleanup_logs
    echo ""
    
    cleanup_checkpoints
    echo ""
    
    cleanup_onnx_files
    echo ""
    
    cleanup_temp_files
    echo ""
    
    generate_disk_report
    echo ""
    
    # 완료 시간
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "유지보수 완료 (${duration}초 소요)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        log_info "실제 정리를 실행하려면 -n 옵션 없이 다시 실행하세요."
    fi
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi