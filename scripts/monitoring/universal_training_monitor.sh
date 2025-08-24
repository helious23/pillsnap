#!/bin/bash
# PillSnap Universal Training Monitor
# 모든 Stage(1-4) 학습 과정을 실시간으로 모니터링하는 통합 스크립트
#
# 기능:
# - 실시간 학습 로그 출력  
# - GPU 상태 및 메모리 사용량
# - 프로세스 상태 및 리소스 사용률
# - Stage별 진행상황 및 성능 지표
# - 자동 Stage 감지 및 맞춤형 정보 표시
#
# 사용법:
#   ./scripts/monitoring/universal_training_monitor.sh
#   ./scripts/monitoring/universal_training_monitor.sh --stage 2
#   ./scripts/monitoring/universal_training_monitor.sh --help

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 기본 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REFRESH_INTERVAL=2
SPECIFIED_STAGE=""
LOG_LINES=10

# 도움말 표시
show_help() {
    echo -e "${CYAN}PillSnap Universal Training Monitor${NC}"
    echo "==============================================="
    echo
    echo -e "${YELLOW}사용법:${NC}"
    echo "  $0 [옵션]"
    echo
    echo -e "${YELLOW}옵션:${NC}"
    echo "  --stage N     특정 Stage(1-4) 모니터링"
    echo "  --interval N  새로고침 간격(초, 기본값: 2)"
    echo "  --lines N     표시할 로그 라인 수(기본값: 10)"
    echo "  --help        이 도움말 표시"
    echo
    echo -e "${YELLOW}예시:${NC}"
    echo "  $0                    # 자동 감지 모니터링"
    echo "  $0 --stage 2         # Stage 2 전용 모니터링"
    echo "  $0 --interval 1      # 1초마다 새로고침"
    echo
}

# 명령행 인수 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            SPECIFIED_STAGE="$2"
            shift 2
            ;;
        --interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --lines)
            LOG_LINES="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 알 수 없는 옵션: $1${NC}"
            echo "사용법을 보려면 --help를 사용하세요."
            exit 1
            ;;
    esac
done

# 현재 학습 프로세스 감지
detect_training_process() {
    local pids=($(ps aux | grep -E "(train_classification_stage|train\.py|training\.py)" | grep -v grep | awk '{print $2}'))
    
    if [ ${#pids[@]} -eq 0 ]; then
        return 1
    fi
    
    echo "${pids[0]}"  # 첫 번째 PID 반환
}

# Stage 자동 감지
detect_current_stage() {
    local pid=$1
    
    # 프로세스 명령어에서 --stage 파라미터 찾기
    local stage=$(ps -p "$pid" -o args --no-headers | grep -oE -- '--stage [0-9]+' | grep -oE '[0-9]+' | head -1)
    
    if [ -n "$stage" ]; then
        echo "$stage"
        return
    fi
    
    # 최근 모델 파일에서 추정
    local models_dir="$PROJECT_ROOT/artifacts/models/classification"
    if [ -d "$models_dir" ]; then
        if [ -f "$models_dir/best_classifier_4523classes.pt" ]; then
            echo "4"
        elif [ -f "$models_dir/best_classifier_1000classes.pt" ]; then
            echo "3"
        elif [ -f "$models_dir/best_classifier_250classes.pt" ]; then
            echo "2"
        elif [ -f "$models_dir/best_classifier_50classes.pt" ]; then
            echo "1"
        else
            echo "?"
        fi
    else
        echo "?"
    fi
}

# GPU 정보 가져오기
get_gpu_info() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "GPU: N/A (nvidia-smi 없음)"
        return
    fi
    
    local gpu_info=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$gpu_info" ]; then
        echo "$gpu_info" | while IFS=',' read -r name util mem_used mem_total temp power; do
            name=$(echo "$name" | xargs)
            util=$(echo "$util" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            temp=$(echo "$temp" | xargs)
            power=$(echo "$power" | xargs)
            
            local mem_percent=$((mem_used * 100 / mem_total))
            
            printf "%-25s │ %3s%% │ %5s/%5sMB (%2s%%) │ %2s°C │ %3sW" \
                "$name" "$util" "$mem_used" "$mem_total" "$mem_percent" "$temp" "$power"
        done
    else
        echo "GPU: 정보 없음"
    fi
}

# 프로세스 정보 가져오기
get_process_info() {
    local pid=$1
    
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "프로세스: 종료됨 (PID: $pid)"
        return
    fi
    
    local proc_info=$(ps -p "$pid" -o pid,pcpu,pmem,etime,args --no-headers)
    echo "$proc_info" | while read -r p_pid p_cpu p_mem p_time p_args; do
        printf "PID: %-6s │ CPU: %5s%% │ MEM: %5s%% │ 시간: %-10s" \
            "$p_pid" "$p_cpu" "$p_mem" "$p_time"
    done
}

# Stage별 정보 가져오기
get_stage_info() {
    local stage=$1
    
    case "$stage" in
        1)
            echo "Stage 1: 기본 파이프라인 검증 (5K 샘플, 50 클래스)"
            echo "목표: 40% 정확도, GPU 메모리 < 14GB"
            ;;
        2)
            echo "Stage 2: 성능 기준선 확립 (25K 샘플, 250 클래스)"
            echo "목표: 60% 정확도, 스케일링 최적화"
            ;;
        3)
            echo "Stage 3: 확장성 테스트 (100K 샘플, 1K 클래스)"
            echo "목표: 85% 정확도, 대용량 데이터 처리"
            ;;
        4)
            echo "Stage 4: 프로덕션 배포 (500K 샘플, 4.5K 클래스)"
            echo "목표: 92% 정확도, 전체 데이터셋"
            ;;
        *)
            echo "Stage ?: 자동 감지 실패"
            echo "수동으로 --stage 옵션을 지정해보세요"
            ;;
    esac
}

# 최신 로그 가져오기
get_recent_logs() {
    local lines=$1
    
    # 다양한 로그 경로 시도
    local log_paths=(
        "/home/max16/pillsnap_data/exp/exp01/logs/src.training.train_stage3_two_stage_*.log"
        "/home/max16/pillsnap_data/exp/exp01/logs/__main___*.log"
        "/tmp/pillsnap_training_stage*/training.log"
        "/tmp/pillsnap_training/training.log"
        "/tmp/training*.log"
        "$PROJECT_ROOT/logs/training*.log"
        "$PROJECT_ROOT/*.log"
    )
    
    local found_log=""
    for pattern in "${log_paths[@]}"; do
        for log_file in $pattern; do
            if [ -f "$log_file" ]; then
                found_log="$log_file"
                break 2
            fi
        done
    done
    
    if [ -n "$found_log" ]; then
        echo -e "${CYAN}📝 최신 로그 ($found_log):${NC}"
        tail -n "$lines" "$found_log" | grep -E "(Epoch|Batch|Loss|Acc|INFO|ERROR|WARNING|완료|달성)" | tail -5
    else
        echo -e "${YELLOW}📝 로그 파일을 찾을 수 없음 (프로세스 출력만 표시)${NC}"
    fi
}

# 성능 지표 가져오기
get_performance_metrics() {
    local stage=$1
    
    local models_dir="$PROJECT_ROOT/artifacts/models/classification"
    local model_file=""
    
    case "$stage" in
        1) model_file="$models_dir/best_classifier_50classes.pt" ;;
        2) model_file="$models_dir/best_classifier_250classes.pt" ;;
        3) model_file="$models_dir/best_classifier_1000classes.pt" ;;
        4) model_file="$models_dir/best_classifier_4523classes.pt" ;;
    esac
    
    if [ -f "$model_file" ]; then
        echo -e "${GREEN}✅ Stage $stage 모델 존재${NC}"
        # Python으로 모델에서 정확도 추출 시도
        local accuracy=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$model_file', map_location='cpu')
    acc = checkpoint.get('best_accuracy', 0)
    print(f'{acc:.1%}')
except:
    print('N/A')
" 2>/dev/null || echo "N/A")
        echo "최고 정확도: $accuracy"
    else
        echo -e "${YELLOW}⏳ Stage $stage 모델 아직 없음${NC}"
    fi
}

# 메인 모니터링 루프
main_monitor() {
    echo -e "${CYAN}"
    echo "██████╗ ██╗██╗     ██╗     ███████╗███╗   ██╗ █████╗ ██████╗ "
    echo "██╔══██╗██║██║     ██║     ██╔════╝████╗  ██║██╔══██╗██╔══██╗"
    echo "██████╔╝██║██║     ██║     ███████╗██╔██╗ ██║███████║██████╔╝"
    echo "██╔═══╝ ██║██║     ██║     ╚════██║██║╚██╗██║██╔══██║██╔═══╝ "
    echo "██║     ██║███████╗███████╗███████║██║ ╚████║██║  ██║██║     "
    echo "╚═╝     ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     "
    echo -e "${NC}"
    echo -e "${PURPLE}Universal Training Monitor${NC}"
    echo "================================================"
    echo
    
    while true; do
        clear
        echo -e "${CYAN}🚀 PillSnap Universal Training Monitor${NC}"
        echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo "================================================"
        
        # 프로세스 감지
        local training_pid=$(detect_training_process)
        
        if [ -z "$training_pid" ]; then
            echo -e "${RED}❌ 학습 프로세스를 찾을 수 없습니다${NC}"
            echo
            echo -e "${YELLOW}학습을 시작하려면:${NC}"
            echo "source .venv/bin/activate"
            echo "python -m src.training.train_classification_stage --stage N --epochs 30"
            echo
            echo -e "${PURPLE}모니터링을 계속하려면 Ctrl+C 후 프로세스 시작 후 다시 실행하세요${NC}"
            sleep 5
            continue
        fi
        
        # Stage 감지
        local current_stage="${SPECIFIED_STAGE:-$(detect_current_stage $training_pid)}"
        
        echo -e "${GREEN}✅ 학습 프로세스 감지됨${NC}"
        echo
        
        # Stage 정보
        echo -e "${PURPLE}📊 STAGE 정보${NC}"
        echo "────────────────────────────────────────────────"
        get_stage_info "$current_stage"
        echo
        
        # GPU 정보
        echo -e "${PURPLE}🎮 GPU 상태${NC}"
        echo "────────────────────────────────────────────────"
        printf "%-25s │ 사용률 │ %-18s │ 온도 │ 전력\n" "GPU" "메모리"
        echo "────────────────────────────────────────────────"
        get_gpu_info
        echo
        echo
        
        # 프로세스 정보
        echo -e "${PURPLE}💻 프로세스 상태${NC}"
        echo "────────────────────────────────────────────────"
        get_process_info "$training_pid"
        echo
        echo
        
        # 성능 지표
        echo -e "${PURPLE}📈 성능 지표${NC}"
        echo "────────────────────────────────────────────────"
        get_performance_metrics "$current_stage"
        echo
        
        # 최신 로그
        echo -e "${PURPLE}📝 실시간 로그${NC}"
        echo "────────────────────────────────────────────────"
        get_recent_logs "$LOG_LINES"
        echo
        
        echo "────────────────────────────────────────────────"
        echo -e "${YELLOW}새로고침: ${REFRESH_INTERVAL}초 │ 종료: Ctrl+C${NC}"
        
        sleep "$REFRESH_INTERVAL"
    done
}

# 의존성 확인
check_dependencies() {
    local missing_deps=()
    
    if ! command -v ps &> /dev/null; then
        missing_deps+=("ps")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}❌ 필수 의존성이 없습니다: ${missing_deps[*]}${NC}"
        exit 1
    fi
}

# 메인 실행
main() {
    echo -e "${CYAN}PillSnap Universal Training Monitor 시작...${NC}"
    
    # 의존성 확인
    check_dependencies
    
    # 인터럽트 핸들링
    trap 'echo -e "\n${YELLOW}🛑 모니터링 종료됨${NC}"; exit 0' INT TERM
    
    # 메인 모니터링 시작
    main_monitor
}

# 스크립트가 직접 실행될 때만 main 함수 호출
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi