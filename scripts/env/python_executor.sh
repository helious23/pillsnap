#!/bin/bash
# 안전한 Python 실행 래퍼 스크립트
# 목적: 항상 Python 3.11.13 가상환경으로만 실행되도록 강제

set -euo pipefail

# 가상환경 Python 경로 (절대 경로로 고정)
VENV_PYTHON="/home/max16/pillsnap/.venv/bin/python"
PROJECT_ROOT="/home/max16/pillsnap"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# 가상환경 Python 존재 확인
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ ERROR: 가상환경 Python을 찾을 수 없습니다: $VENV_PYTHON${NC}"
    exit 1
fi

# 프로젝트 루트로 이동
cd "$PROJECT_ROOT"

# 환경변수 설정
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset"

# Python 3.11.13으로 실행
echo -e "${BLUE}🐍 Python 3.11.13 실행: $*${NC}"
exec "$VENV_PYTHON" "$@"