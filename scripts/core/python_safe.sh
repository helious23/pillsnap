#!/bin/bash
# 안전한 Python 실행 래퍼
# 사용법: ./scripts/core/python_safe.sh [python 명령어와 인수들]

set -euo pipefail

# 가상환경 Python 경로
VENV_PYTHON="/home/max16/pillsnap/.venv/bin/python"
PROJECT_ROOT="/home/max16/pillsnap"

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 가상환경 확인
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ 가상환경 Python을 찾을 수 없습니다: $VENV_PYTHON${NC}"
    exit 1
fi

# 프로젝트 루트로 이동
cd "$PROJECT_ROOT"

# 환경변수 설정
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PILLSNAP_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"

# Python 실행
echo -e "${GREEN}🐍 가상환경 Python 실행: $*${NC}"
exec "$VENV_PYTHON" "$@"