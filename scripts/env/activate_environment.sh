#!/bin/bash
# PillSnap ML 프로젝트 전용 가상환경 활성화 스크립트
# 목적: Python 3.11.13 가상환경 강제 사용, 시스템 Python 3.12+ 회피

set -euo pipefail

# 프로젝트 루트 경로 (절대 경로로 고정)
PROJECT_ROOT="/home/max16/pillsnap"
VENV_PATH="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_PATH/bin/python"

# 색상 정의 (콘솔 가독성)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== PillSnap ML 가상환경 활성화 ===${NC}"

# 1) 프로젝트 루트로 이동
if [ ! -d "$PROJECT_ROOT" ]; then
    echo -e "${RED}❌ ERROR: 프로젝트 루트를 찾을 수 없습니다: $PROJECT_ROOT${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"
echo -e "${GREEN}📁 프로젝트 루트로 이동: $PROJECT_ROOT${NC}"

# 2) 가상환경 존재 확인
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ ERROR: 가상환경을 찾을 수 없습니다: $VENV_PYTHON${NC}"
    echo -e "${YELLOW}💡 TIP: .venv 디렉토리가 올바르게 설정되었는지 확인하세요${NC}"
    exit 1
fi

# 3) Python 버전 검증 (3.11.x 강제)
PYTHON_VERSION=$("$VENV_PYTHON" --version 2>&1 | grep -o "3\.11\.[0-9]*" || echo "")
if [ -z "$PYTHON_VERSION" ]; then
    echo -e "${RED}❌ ERROR: Python 3.11.x 버전이 아닙니다${NC}"
    echo -e "${YELLOW}현재 버전: $($VENV_PYTHON --version)${NC}"
    echo -e "${YELLOW}필요 버전: Python 3.11.x${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 버전 확인: $PYTHON_VERSION${NC}"

# 4) PyTorch + CUDA 검증
echo -e "${BLUE}🔥 GPU 환경 검증 중...${NC}"
GPU_STATUS=$("$VENV_PYTHON" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU수: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU이름: {torch.cuda.get_device_name(0)}')
" 2>/dev/null || echo "GPU 검증 실패")

if [[ "$GPU_STATUS" == *"CUDA: True"* ]]; then
    echo -e "${GREEN}✅ GPU 환경 정상${NC}"
    echo "$GPU_STATUS" | while read line; do echo -e "${GREEN}  $line${NC}"; done
else
    echo -e "${RED}❌ WARNING: GPU 환경 문제${NC}"
    echo "$GPU_STATUS"
fi

# 5) 가상환경 활성화
echo -e "${BLUE}🚀 가상환경 활성화 중...${NC}"
source "$VENV_PATH/bin/activate"

# 6) 환경변수 설정
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset"

# 7) 활성화 완료 메시지
echo -e "${GREEN}✅ 가상환경 활성화 완료!${NC}"
echo -e "${YELLOW}📋 환경 정보:${NC}"
echo -e "${YELLOW}  - Python: $PYTHON_VERSION${NC}"
echo -e "${YELLOW}  - 프로젝트: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}  - 데이터: $PILLSNAP_DATA_ROOT${NC}"
echo ""
echo -e "${BLUE}💻 사용법:${NC}"
echo -e "${BLUE}  source scripts/activate_env.sh    # 환경 활성화${NC}"
echo -e "${BLUE}  python [스크립트]                 # Python 실행${NC}"
echo -e "${BLUE}  deactivate                        # 환경 비활성화${NC}"
echo ""

# 8) 프롬프트 변경 (활성 상태 표시)
export PS1="(pillsnap-py3.11) \u@\h:\w\$ "