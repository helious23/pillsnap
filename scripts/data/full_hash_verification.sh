#!/bin/bash
# PillSnap ML 전체 해시 검증 스크립트
# AMD Ryzen 7 7800X3D + 128GB RAM 최적화

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 시스템 정보 출력
show_system_info() {
    log "=== 시스템 정보 ==="
    echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "코어/스레드: $(nproc) threads"
    echo "메모리: $(free -h | grep Mem | awk '{print $2}')"
    echo "현재 시간: $(date)"
    echo ""
}

# 전처리 - 파일 목록 생성
generate_file_lists() {
    log "파일 목록 생성 중..."
    
    # Windows SSD 파일 목록
    find /mnt/windows/pillsnap_data/ -name "*.png" -type f > /tmp/windows_files.txt &
    WIN_PID=$!
    
    # Linux SSD 이미지 파일 목록
    find dataset/train/images/ -name "*.png" -type f > /tmp/linux_images.txt &
    LINUX_IMG_PID=$!
    
    # Linux SSD 라벨 파일 목록
    find dataset/train/labels/ -name "*.json" -type f > /tmp/linux_labels.txt &
    LINUX_LBL_PID=$!
    
    # 외장 HDD 파일 목록 (검증용)
    find /mnt/external/pillsnap_dataset/data/ -name "*.png" -o -name "*.json" > /tmp/external_files.txt &
    EXT_PID=$!
    
    # 모든 백그라운드 작업 완료 대기
    wait $WIN_PID $LINUX_IMG_PID $LINUX_LBL_PID $EXT_PID
    
    # 통합 파일 목록 생성
    cat /tmp/windows_files.txt /tmp/linux_images.txt /tmp/linux_labels.txt > /tmp/all_local_files.txt
    
    local win_count=$(wc -l < /tmp/windows_files.txt)
    local linux_img_count=$(wc -l < /tmp/linux_images.txt)
    local linux_lbl_count=$(wc -l < /tmp/linux_labels.txt)
    local ext_count=$(wc -l < /tmp/external_files.txt)
    local total_local=$(wc -l < /tmp/all_local_files.txt)
    
    log "파일 개수 확인:"
    echo "  Windows SSD: ${win_count}개"
    echo "  Linux SSD (images): ${linux_img_count}개"
    echo "  Linux SSD (labels): ${linux_lbl_count}개"
    echo "  로컬 총합: ${total_local}개"
    echo "  외장 HDD: ${ext_count}개"
    echo ""
}

# 병렬 해시 계산 (최대 성능)
parallel_hash_calculation() {
    log "병렬 해시 계산 시작 (최대 성능 모드)"
    
    # 워커 수 결정 (스레드당 2-3개)
    local workers=40
    local chunk_size=8
    
    log "설정: ${workers}개 워커, 청크 크기 ${chunk_size}"
    
    # 시작 시간 기록
    local start_time=$(date +%s)
    
    # 병렬 해시 계산 (진행률 표시)
    cat /tmp/all_local_files.txt | \
        xargs -P ${workers} -n ${chunk_size} -I {} sh -c '
            for file in "$@"; do
                if [[ -f "$file" ]]; then
                    md5sum "$file" 2>/dev/null || echo "ERROR: $file"
                fi
            done
        ' -- {} > /tmp/local_hashes.txt 2>&1
    
    # 종료 시간 및 소요 시간 계산
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    success "해시 계산 완료! 소요 시간: ${minutes}분 ${seconds}초"
    
    # 결과 통계
    local hash_count=$(grep -v "ERROR:" /tmp/local_hashes.txt | wc -l)
    local error_count=$(grep "ERROR:" /tmp/local_hashes.txt | wc -l)
    
    echo "  성공: ${hash_count}개"
    if [[ $error_count -gt 0 ]]; then
        warning "오류: ${error_count}개"
    fi
    echo ""
}

# 해시 비교 및 검증
verify_hashes() {
    log "해시 검증 시작..."
    
    # 외장 HDD 해시 계산 (비교용)
    log "외장 HDD 해시 계산 중..."
    find /mnt/external/pillsnap_dataset/data/ \( -name "*.png" -o -name "*.json" \) -type f | \
        xargs -P 20 -n 10 md5sum > /tmp/external_hashes.txt 2>/dev/null
    
    # 해시 비교
    local total_files=0
    local matched=0
    local mismatched=0
    local missing=0
    
    while IFS= read -r line; do
        if [[ $line =~ ^([a-f0-9]+)\ \ (.+)$ ]]; then
            local hash="${BASH_REMATCH[1]}"
            local filepath="${BASH_REMATCH[2]}"
            local filename=$(basename "$filepath")
            
            # 로컬에서 동일한 파일의 해시 찾기
            local local_hash=$(grep " ${filename}$" /tmp/local_hashes.txt | head -1 | cut -d' ' -f1)
            
            total_files=$((total_files + 1))
            
            if [[ -n "$local_hash" ]]; then
                if [[ "$hash" == "$local_hash" ]]; then
                    matched=$((matched + 1))
                else
                    mismatched=$((mismatched + 1))
                    echo "MISMATCH: $filename" >> /tmp/hash_mismatches.txt
                fi
            else
                missing=$((missing + 1))
                echo "MISSING: $filename" >> /tmp/hash_missing.txt
            fi
            
            # 진행률 표시 (1000개마다)
            if [[ $((total_files % 1000)) -eq 0 ]]; then
                echo -ne "\r검증 진행률: ${total_files}개 처리됨"
            fi
        fi
    done < /tmp/external_hashes.txt
    
    echo "" # 줄바꿈
    
    # 결과 출력
    log "=== 해시 검증 결과 ==="
    echo "  총 파일: ${total_files}개"
    echo "  일치: ${matched}개 ($(( matched * 100 / total_files ))%)"
    
    if [[ $mismatched -gt 0 ]]; then
        error "불일치: ${mismatched}개"
        echo "    상세: /tmp/hash_mismatches.txt 참조"
    fi
    
    if [[ $missing -gt 0 ]]; then
        warning "누락: ${missing}개"
        echo "    상세: /tmp/hash_missing.txt 참조"
    fi
    
    if [[ $mismatched -eq 0 && $missing -eq 0 ]]; then
        success "🎉 모든 해시가 완벽하게 일치합니다!"
        return 0
    else
        error "해시 불일치 또는 누락 파일이 있습니다."
        return 1
    fi
}

# 성능 리포트 생성
generate_performance_report() {
    local start_time=$1
    local end_time=$2
    local total_duration=$((end_time - start_time))
    local total_minutes=$((total_duration / 60))
    local total_seconds=$((total_duration % 60))
    
    local total_files=$(wc -l < /tmp/all_local_files.txt)
    local total_size=$(du -ch $(cat /tmp/all_local_files.txt) 2>/dev/null | tail -1 | cut -f1)
    local files_per_sec=$((total_files / total_duration))
    
    log "=== 성능 리포트 ==="
    echo "  총 소요 시간: ${total_minutes}분 ${total_seconds}초"
    echo "  처리된 파일: ${total_files}개"
    echo "  총 데이터 크기: ${total_size}"
    echo "  처리 속도: ${files_per_sec}개/초"
    
    # CPU 사용률 통계
    local cpu_usage=$(grep "cpu " /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}')
    echo "  평균 CPU 사용률: ${cpu_usage}"
    
    # 메모리 사용량
    local mem_info=$(free -h | grep Mem | awk '{print "사용: " $3 " / 전체: " $2}')
    echo "  메모리 ${mem_info}"
    echo ""
}

# 메인 함수
main() {
    log "🚀 PillSnap ML 전체 해시 검증 시작"
    log "AMD Ryzen 7 7800X3D + 128GB RAM 최적화 버전"
    echo ""
    
    # 시작 시간 기록
    local main_start_time=$(date +%s)
    
    # 시스템 정보 출력
    show_system_info
    
    # 작업 디렉토리 확인
    if [[ ! -d "dataset" ]] || [[ ! -d "/mnt/windows/pillsnap_data" ]] || [[ ! -d "/mnt/external/pillsnap_dataset" ]]; then
        error "필요한 디렉토리가 없습니다. 작업 디렉토리를 확인하세요."
        exit 1
    fi
    
    # 1단계: 파일 목록 생성
    generate_file_lists
    
    # 2단계: 병렬 해시 계산
    parallel_hash_calculation
    
    # 3단계: 해시 검증
    if verify_hashes; then
        success "✅ 데이터 무결성 검증 완료!"
        echo ""
        log "모든 데이터가 외장 HDD와 완벽하게 일치합니다."
        log "심볼릭 링크를 안전하게 생성할 수 있습니다."
    else
        error "❌ 데이터 무결성 검증 실패!"
        echo ""
        log "심볼릭 링크 생성 전에 불일치 파일들을 확인하고 수정하세요."
        exit 1
    fi
    
    # 종료 시간 및 성능 리포트
    local main_end_time=$(date +%s)
    generate_performance_report $main_start_time $main_end_time
    
    success "🎉 전체 해시 검증 완료!"
}

# 스크립트 실행
main "$@"