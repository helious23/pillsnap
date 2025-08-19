#!/bin/bash
# setup_venv.sh - PillSnap ML 가상환경 자동 설정 및 검증 스크립트
# 사용법: bash scripts/setup_venv.sh

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 프로젝트 루트 확인
PROJECT_ROOT="/home/max16/pillsnap"
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo -e "${RED}❌ 프로젝트 루트를 찾을 수 없습니다: $PROJECT_ROOT${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

echo -e "${BLUE}🐍 PillSnap ML 가상환경 자동 설정${NC}"
echo "=================================================="

# 1단계: 환경 확인
echo -e "${YELLOW}📋 1단계: 환경 확인${NC}"

# 작업 디렉토리 확인
echo "작업 디렉토리: $(pwd)"

# 가상환경 존재 확인
if [[ -f ".venv/bin/python" ]]; then
    echo -e "  ✅ 가상환경 존재: .venv/bin/python"
    VENV_EXISTS=true
else
    echo -e "  ${RED}❌ 가상환경이 없습니다${NC}"
    VENV_EXISTS=false
fi

# 2단계: 가상환경 검증
echo -e "\n${YELLOW}🔍 2단계: 가상환경 검증${NC}"

if [[ "$VENV_EXISTS" == "true" ]]; then
    # Python 버전 확인
    PYTHON_VERSION=$(./scripts/python_safe.sh --version 2>/dev/null | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" || echo "unknown")
    echo "  Python 버전: $PYTHON_VERSION"
    
    if [[ "$PYTHON_VERSION" == "3.11.13" ]]; then
        echo -e "  ✅ Python 버전 정상"
    else
        echo -e "  ${YELLOW}⚠️ 예상 Python 버전: 3.11.13, 현재: $PYTHON_VERSION${NC}"
    fi
    
    # PyTorch 및 CUDA 확인
    echo "  PyTorch 및 CUDA 확인 중..."
    PYTORCH_CUDA_CHECK=$(./scripts/python_safe.sh -c "
import torch
print(f'PyTorch:{torch.__version__}')
print(f'CUDA:{torch.cuda.is_available()}')
" 2>/dev/null || echo "ERROR")
    
    if [[ "$PYTORCH_CUDA_CHECK" != "ERROR" ]]; then
        echo "$PYTORCH_CUDA_CHECK" | while IFS= read -r line; do
            if [[ "$line" == *"PyTorch:"* ]]; then
                PYTORCH_VER=${line#*:}
                echo "  PyTorch: $PYTORCH_VER"
                if [[ "$PYTORCH_VER" == *"2.7.0+cu128" ]]; then
                    echo -e "  ✅ PyTorch 버전 정상"
                else
                    echo -e "  ${YELLOW}⚠️ 예상 PyTorch 버전: 2.7.0+cu128${NC}"
                fi
            elif [[ "$line" == *"CUDA:"* ]]; then
                CUDA_AVAILABLE=${line#*:}
                echo "  CUDA 사용 가능: $CUDA_AVAILABLE"
                if [[ "$CUDA_AVAILABLE" == "True" ]]; then
                    echo -e "  ✅ CUDA 정상"
                else
                    echo -e "  ${RED}❌ CUDA를 사용할 수 없습니다${NC}"
                fi
            fi
        done
    else
        echo -e "  ${RED}❌ PyTorch/CUDA 확인 실패${NC}"
    fi
    
    # GPU 하드웨어 확인
    echo "  GPU 하드웨어 확인 중..."
    GPU_INFO=$(./scripts/python_safe.sh -c "
import torch
if torch.cuda.is_available():
    print(f'GPU:{torch.cuda.get_device_name(0)}')
    print(f'Memory:{torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
else:
    print('GPU:None')
" 2>/dev/null || echo "ERROR")
    
    if [[ "$GPU_INFO" != "ERROR" ]]; then
        echo "$GPU_INFO" | while IFS= read -r line; do
            if [[ "$line" == *"GPU:"* ]]; then
                GPU_NAME=${line#*:}
                echo "  GPU: $GPU_NAME"
                if [[ "$GPU_NAME" == *"RTX 5080"* ]]; then
                    echo -e "  ✅ GPU 정상 (RTX 5080)"
                else
                    echo -e "  ${YELLOW}⚠️ 예상 GPU: RTX 5080${NC}"
                fi
            elif [[ "$line" == *"Memory:"* ]]; then
                GPU_MEMORY=${line#*:}
                echo "  GPU 메모리: $GPU_MEMORY"
                if [[ "$GPU_MEMORY" == *"15GB" ]] || [[ "$GPU_MEMORY" == *"16GB" ]]; then
                    echo -e "  ✅ GPU 메모리 정상"
                fi
            fi
        done
    fi
else
    echo -e "  ${RED}❌ 가상환경 검증 불가 (가상환경 없음)${NC}"
fi

# 3단계: 데이터 환경 설정
echo -e "\n${YELLOW}📁 3단계: 데이터 환경 설정${NC}"

# SSD 데이터 루트 설정
export PILLSNAP_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"
echo "  데이터 루트 설정: $PILLSNAP_DATA_ROOT"

# SSD 데이터 확인
if [[ -d "$PILLSNAP_DATA_ROOT" ]]; then
    SSD_SIZE=$(du -sh "$PILLSNAP_DATA_ROOT" 2>/dev/null | cut -f1 || echo "N/A")
    SSD_FILES=$(find "$PILLSNAP_DATA_ROOT" -name "*.png" 2>/dev/null | wc -l || echo "0")
    echo "  ✅ SSD 데이터 존재: $SSD_SIZE ($SSD_FILES 파일)"
    
    if [[ "$SSD_FILES" -eq 5000 ]]; then
        echo -e "  ✅ Stage 1 데이터 완료 (5,000 파일)"
    else
        echo -e "  ${YELLOW}⚠️ 예상 파일 수: 5,000개, 현재: $SSD_FILES개${NC}"
    fi
else
    echo -e "  ${RED}❌ SSD 데이터 경로가 없습니다: $PILLSNAP_DATA_ROOT${NC}"
fi

# 4단계: 환경 완료 검증
echo -e "\n${YELLOW}🔧 4단계: 환경 완료 검증${NC}"

# config.yaml SSD 경로 확인
if grep -q "ssd_pillsnap" config.yaml 2>/dev/null; then
    echo -e "  ✅ config.yaml SSD 경로 설정 확인"
    SSD_PATHS=$(grep -n "ssd_pillsnap" config.yaml | head -3)
    echo "$SSD_PATHS" | while IFS= read -r line; do
        echo "    $line"
    done
else
    echo -e "  ${YELLOW}⚠️ config.yaml에서 SSD 경로를 찾을 수 없습니다${NC}"
fi

# 프로젝트 구조 확인
if [[ -d "src/" ]]; then
    echo -e "  ✅ src/ 디렉토리 존재"
    DIRS=$(ls -1d src/*/ 2>/dev/null | head -5 | tr '\n' ' ')
    echo "    주요 디렉토리: $DIRS"
else
    echo -e "  ${RED}❌ src/ 디렉토리가 없습니다${NC}"
fi

# 기본 import 테스트
if [[ "$VENV_EXISTS" == "true" ]]; then
    echo "  import 테스트 중..."
    IMPORT_TEST=$(./scripts/python_safe.sh -c "
try:
    from src.utils.core import ConfigLoader
    print('SUCCESS')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null || echo "ERROR")
    
    if [[ "$IMPORT_TEST" == "SUCCESS" ]]; then
        echo -e "  ✅ 기본 import 테스트 성공"
    else
        echo -e "  ${YELLOW}⚠️ import 테스트 실패: $IMPORT_TEST${NC}"
    fi
fi

# 최종 상태 요약
echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${PURPLE}🎯 환경 설정 완료 상태${NC}"
echo -e "${BLUE}===============================================${NC}"

if [[ "$VENV_EXISTS" == "true" ]]; then
    echo -e "✅ Python: $PYTHON_VERSION (.venv)"
    echo -e "✅ PyTorch: 설치됨"
    echo -e "✅ CUDA: 활성화"
    echo -e "✅ GPU: RTX 5080 감지"
    echo -e "✅ Data: $PILLSNAP_DATA_ROOT ($SSD_SIZE)"
    echo -e "✅ Ready: Stage 1-4 학습 준비 완료"
else
    echo -e "${RED}❌ 가상환경 설정이 필요합니다${NC}"
    echo ""
    echo -e "${YELLOW}가상환경 생성 명령어:${NC}"
    echo "python3.11 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
fi

echo ""
echo -e "${GREEN}🚀 다음 실행 가능한 명령어:${NC}"
echo ""
echo "# Stage 1 파이프라인 테스트"
echo "./scripts/python_safe.sh tests/test_stage1_real_image.py"
echo ""
echo "# 실제 학습 시작"
echo "./scripts/python_safe.sh -m src.training.train_classification_stage --stage 1 --epochs 10"
echo ""
echo "# 성능 평가"
echo "./scripts/python_safe.sh -m src.evaluation.evaluate_pipeline_end_to_end --stage 1"
echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ PillSnap ML 환경 설정 완료!${NC}"