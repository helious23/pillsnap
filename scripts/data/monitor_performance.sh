#!/bin/bash
# 실시간 성능 모니터링 스크립트
# 해시 검증 실행 중 별도 터미널에서 실행

MONITOR_INTERVAL=5
LOG_FILE="/tmp/hash_verification_performance.log"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 초기화
echo "Performance Monitoring Started: $(date)" > "$LOG_FILE"

while true; do
    clear
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  PillSnap ML 해시 검증 성능 모니터  ${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "${BLUE}$(date)${NC}"
    echo ""
    
    # CPU 사용률 (코어별)
    echo -e "${GREEN}📊 CPU 사용률${NC}"
    top -bn1 | grep "Cpu" | head -1 | sed 's/Cpu(s)://' | awk '{
        printf "  사용률: %.1f%%  대기: %.1f%%  시스템: %.1f%%\n", 
        $1+$3, $4, $2
    }'
    
    # 개별 코어 사용률 (간단히)
    echo -e "${GREEN}🔥 코어별 부하${NC}"
    if command -v mpstat >/dev/null 2>&1; then
        mpstat -P ALL 1 1 2>/dev/null | grep -v Average | grep -E '^[0-9]' | grep -v ' all ' | head -8 | awk '{
            if(NF >= 11) {
                usage = 100 - $NF
                printf "  Core %2s: %5.1f%%\n", $2, usage
            }
        }'
    else
        echo "  mpstat 없음 - sysstat 패키지 설치 필요"
    fi
    
    # 메모리 사용량
    echo -e "${GREEN}💾 메모리 사용량${NC}"
    free -h | grep Mem | awk '{
        printf "  사용: %s / %s (%s 사용 가능)\n", $3, $2, $7
    }'
    
    # 디스크 I/O
    echo -e "${GREEN}💽 디스크 I/O${NC}"
    iostat -x 1 1 2>/dev/null | grep -E "nvme|sda" | head -3 | awk '{
        if(NF > 5) printf "  %s: Read %.1f MB/s, Write %.1f MB/s\n", 
        $1, $6/1024, $7/1024
    }'
    
    # 진행 중인 md5sum 프로세스 수
    echo -e "${GREEN}⚡ 활성 프로세스${NC}"
    md5_count=$(pgrep -c md5sum 2>/dev/null || echo "0")
    xargs_count=$(pgrep -c xargs 2>/dev/null || echo "0")
    find_count=$(pgrep -c find 2>/dev/null || echo "0")
    
    printf "  md5sum 프로세스: %s개\n" "${md5_count}"
    printf "  xargs 프로세스: %s개\n" "${xargs_count}" 
    printf "  find 프로세스: %s개\n" "${find_count}"
    
    # 부하 평균
    echo -e "${GREEN}📈 시스템 부하${NC}"
    uptime | awk -F'load average:' '{print "  Load Average:" $2}'
    
    # 해시 파일 진행률 (추정)
    if [[ -f "/tmp/local_hashes.txt" ]]; then
        hash_lines=$(wc -l < /tmp/local_hashes.txt 2>/dev/null || echo "0")
        echo -e "${GREEN}🔍 해시 계산 진행${NC}"
        echo "  완료된 해시: ${hash_lines}개"
        
        # 에러 개수
        error_count=$(grep -c "ERROR:" /tmp/local_hashes.txt 2>/dev/null || echo "0")
        error_count=$(echo "$error_count" | tr -d '\n\r')
        if [[ "$error_count" -gt 0 ]] 2>/dev/null; then
            echo -e "  ${RED}에러: ${error_count}개${NC}"
        fi
    fi
    
    # 온도 정보 (가능하다면)
    if command -v sensors >/dev/null 2>&1; then
        echo -e "${GREEN}🌡️  시스템 온도${NC}"
        sensors 2>/dev/null | grep -E "Core|Tctl" | head -4 | while read line; do
            echo "  $line"
        done
    fi
    
    # 로그에 기록
    {
        echo "$(date),$(top -bn1 | grep "Cpu" | awk '{print $2+$4}'),$(free | grep Mem | awk '{print $3/$2*100}'),${md5_count}"
    } >> "$LOG_FILE"
    
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    echo -e "${YELLOW}로그: $LOG_FILE${NC}"
    
    sleep $MONITOR_INTERVAL
done