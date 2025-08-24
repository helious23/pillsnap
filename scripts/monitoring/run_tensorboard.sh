#!/bin/bash

# TensorBoard 실행 스크립트
# PillSnap ML 학습 모니터링

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본 설정
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/home/max16/pillsnap/artifacts/tensorboard}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"
RELOAD_INTERVAL="${RELOAD_INTERVAL:-5}"

# 헬프 메시지
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dir DIR      TensorBoard 로그 디렉토리 (기본: $TENSORBOARD_DIR)"
    echo "  -p, --port PORT    포트 번호 (기본: $PORT)"
    echo "  -h, --host HOST    호스트 주소 (기본: $HOST)"
    echo "  --reload           자동 새로고침 모드"
    echo "  --help             이 도움말 표시"
    echo ""
    echo "Examples:"
    echo "  $0                                    # 기본 설정으로 실행"
    echo "  $0 -p 6007                           # 포트 6007로 실행"
    echo "  $0 -d runs/experiment_1              # 특정 디렉토리 모니터링"
    echo "  $0 --reload                          # 자동 새로고침 모드"
}

# 인자 파싱
RELOAD_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            TENSORBOARD_DIR="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        --reload)
            RELOAD_FLAG="--reload_interval $RELOAD_INTERVAL"
            shift
            ;;
        -r|--reload-interval)
            RELOAD_INTERVAL="$2"
            RELOAD_FLAG="--reload_interval $RELOAD_INTERVAL"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 PillSnap ML TensorBoard Monitor${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Python 가상환경 활성화
if [ -f "/home/max16/pillsnap/.venv/bin/activate" ]; then
    source /home/max16/pillsnap/.venv/bin/activate
    echo -e "${GREEN}✅ 가상환경 활성화됨${NC}"
else
    echo -e "${RED}❌ 가상환경을 찾을 수 없습니다${NC}"
    exit 1
fi

# TensorBoard 설치 확인
if ! python -c "import tensorboard" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  TensorBoard가 설치되지 않았습니다. 설치 중...${NC}"
    pip install tensorboard
fi

# 로그 디렉토리 확인
if [ ! -d "$TENSORBOARD_DIR" ]; then
    echo -e "${YELLOW}⚠️  로그 디렉토리가 없습니다: $TENSORBOARD_DIR${NC}"
    echo -e "${YELLOW}   디렉토리를 생성합니다...${NC}"
    mkdir -p "$TENSORBOARD_DIR"
fi

# 기존 TensorBoard 프로세스 확인
if lsof -i:$PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  포트 $PORT가 이미 사용 중입니다${NC}"
    echo -e "${YELLOW}   다른 포트를 사용하거나 기존 프로세스를 종료하세요${NC}"
    echo ""
    echo "기존 프로세스 확인:"
    lsof -i:$PORT
    echo ""
    read -p "기존 프로세스를 종료하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "tensorboard.*--port=$PORT"
        sleep 2
        echo -e "${GREEN}✅ 기존 프로세스 종료됨${NC}"
    else
        exit 1
    fi
fi

# TensorBoard 실행
echo ""
echo -e "${GREEN}🚀 TensorBoard 시작 중...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "📁 로그 디렉토리: ${YELLOW}$TENSORBOARD_DIR${NC}"
echo -e "🌐 웹 주소: ${GREEN}http://localhost:$PORT${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}브라우저에서 위 주소로 접속하세요${NC}"
echo -e "${YELLOW}종료하려면 Ctrl+C를 누르세요${NC}"
echo ""

# TensorBoard 실행
tensorboard \
    --logdir="$TENSORBOARD_DIR" \
    --port=$PORT \
    --reload_multifile=true \
    $RELOAD_FLAG \
    --bind_all