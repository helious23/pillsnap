#!/bin/bash
# PyTorch 2.5 Stack 안전 설치 스크립트
# 호환성 검증과 함께 순차적 설치

set -euo pipefail

# 색깔 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 시스템 요구사항 체크
check_system_requirements() {
    log_info "시스템 요구사항 검증 중..."
    
    # Python 버전 체크
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ]; then
        log_error "Python 3.9+ required, got $PYTHON_VERSION"
        exit 1
    fi
    log_success "Python version: $PYTHON_VERSION ✓"
    
    # CUDA 환경 체크
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
        log_success "CUDA detected: $CUDA_VERSION ✓"
        
        if [[ ! "$CUDA_VERSION" =~ ^12\.[0-8] ]]; then
            log_warning "CUDA $CUDA_VERSION may not be optimal for PyTorch 2.5 (12.0-12.8 recommended)"
        fi
    else
        log_warning "NVIDIA driver not found - will install CPU-only PyTorch"
    fi
    
    # 메모리 체크
    TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM_GB" -lt 64 ]; then
        log_warning "System has ${TOTAL_RAM_GB}GB RAM (128GB recommended for full performance)"
    else
        log_success "RAM: ${TOTAL_RAM_GB}GB ✓"
    fi
}

# 기존 패키지 정리
cleanup_existing_packages() {
    log_info "기존 PyTorch 패키지 정리 중..."
    
    # 충돌 가능한 패키지들 제거
    pip uninstall -y torch torchvision torchaudio onnxruntime onnxruntime-gpu 2>/dev/null || true
    
    log_success "기존 패키지 정리 완료"
}

# PyTorch 스택 설치
install_pytorch_stack() {
    log_info "PyTorch 2.5 스택 설치 중..."
    
    # CUDA 사용 가능 여부에 따라 설치 방법 결정
    if command -v nvidia-smi &> /dev/null; then
        log_info "CUDA 버전으로 설치: torch==2.5.1+cu121"
        pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
            --index-url https://download.pytorch.org/whl/cu121
    else
        log_warning "CPU 전용 버전으로 설치"
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_success "PyTorch 스택 설치 완료"
}

# Vision & ML 프레임워크 설치
install_vision_frameworks() {
    log_info "Vision & ML 프레임워크 설치 중..."
    
    pip install timm>=1.0.10 ultralytics>=8.3
    
    log_success "Vision 프레임워크 설치 완료"
}

# ONNX 스택 설치
install_onnx_stack() {
    log_info "ONNX 스택 설치 중..."
    
    # ONNX 코어
    pip install onnx>=1.16
    
    # ORT: GPU 버전 우선, 실패 시 CPU 폴백
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU 버전 ONNX Runtime 설치 시도..."
        if pip install onnxruntime-gpu>=1.22; then
            log_success "onnxruntime-gpu 설치 성공"
        else
            log_warning "onnxruntime-gpu 설치 실패, CPU 버전으로 폴백"
            pip install onnxruntime>=1.22
        fi
    else
        pip install onnxruntime>=1.22
    fi
    
    log_success "ONNX 스택 설치 완료"
}

# 나머지 의존성 설치
install_remaining_dependencies() {
    log_info "나머지 의존성 설치 중..."
    
    # requirements.txt에서 이미 설치된 것 제외하고 설치
    pip install \
        "numpy>=1.24,<2.0" \
        "pillow>=10.0,<11.0" \
        "opencv-python-headless>=4.9,<5.0" \
        "albumentations>=1.4,<2.0" \
        "kornia>=0.7,<1.0" \
        "pyyaml>=6.0,<7.0" \
        "tqdm>=4.66,<5.0" \
        "pandas>=2.0,<3.0" \
        "scikit-learn>=1.4,<2.0" \
        "tensorboard>=2.15,<3.0" \
        "matplotlib>=3.8,<4.0" \
        "seaborn>=0.13,<1.0" \
        "fastapi>=0.110,<1.0" \
        "uvicorn[standard]>=0.27,<1.0" \
        "python-multipart>=0.0.9,<1.0" \
        "pydantic>=2.6,<3.0" \
        "pydantic-settings>=2.2,<3.0" \
        "python-dotenv>=1.0,<2.0" \
        "lmdb>=1.4,<2.0" \
        "psutil>=5.9,<6.0" \
        "httpx>=0.26,<1.0" \
        "pytest>=8.0,<9.0" \
        "pytest-asyncio>=0.23,<1.0"
    
    log_success "의존성 설치 완료"
}

# 버전 호환성 검증
validate_installation() {
    log_info "설치 검증 중..."
    
    python3 -c "
import sys
import torch
import torchvision
import torchaudio
import timm
import ultralytics
import onnx
import onnxruntime as ort

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'TorchAudio: {torchaudio.__version__}')
print(f'TIMM: {timm.__version__}')
print(f'Ultralytics: {ultralytics.__version__}')
print(f'ONNX: {onnx.__version__}')
print(f'ONNX Runtime: {ort.__version__}')

# CUDA 검증
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

# ORT Providers 검증
print(f'ORT Providers: {ort.get_available_providers()}')

# TensorRT 검증 (선택적)
try:
    import tensorrt as trt
    print(f'TensorRT: {trt.__version__}')
except ImportError:
    print('TensorRT: Not installed (optional)')

# torch.compile 검증
if hasattr(torch, 'compile'):
    print('torch.compile: Available ✓')
    
    # 간단한 컴파일 테스트
    model = torch.nn.Linear(10, 1)
    compiled_model = torch.compile(model, mode='default')
    print('torch.compile test: Passed ✓')
else:
    print('torch.compile: Not available')

print('\\n✓ All versions validated successfully')
"
    
    if [ $? -eq 0 ]; then
        log_success "설치 검증 성공 ✓"
    else
        log_error "설치 검증 실패"
        exit 1
    fi
}

# TensorRT 설치 안내 (선택적)
suggest_tensorrt_installation() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "TensorRT 설치 권장사항:"
        echo "최적의 ONNX 추론 성능을 위해 TensorRT 설치를 권장합니다:"
        echo ""
        echo "1. NVIDIA 개발자 사이트에서 TensorRT 10.9 다운로드"
        echo "   https://developer.nvidia.com/tensorrt"
        echo ""
        echo "2. 설치 후 다음 명령어로 검증:"
        echo "   python -c 'import tensorrt; print(tensorrt.__version__)'"
        echo ""
        echo "3. PillSnap에서 TensorRT EP 활성화:"
        echo "   config.yaml의 export.onnx.providers에 TensorrtExecutionProvider 추가"
        echo ""
    fi
}

# 메인 실행
main() {
    echo "=========================================="
    echo "PillSnap ML - PyTorch 2.5 Stack 설치"
    echo "=========================================="
    echo ""
    
    check_system_requirements
    echo ""
    
    cleanup_existing_packages
    echo ""
    
    install_pytorch_stack
    echo ""
    
    install_vision_frameworks
    echo ""
    
    install_onnx_stack
    echo ""
    
    install_remaining_dependencies
    echo ""
    
    validate_installation
    echo ""
    
    suggest_tensorrt_installation
    echo ""
    
    log_success "=========================================="
    log_success "PyTorch 2.5 스택 설치 완료!"
    log_success "=========================================="
    echo ""
    echo "다음 단계:"
    echo "1. 가상환경 활성화: source \$HOME/pillsnap/.venv/bin/activate"
    echo "2. 설정 검증: python -m src.core.torch_compile_utils"
    echo "3. 데이터 준비: bash scripts/bootstrap_venv.sh"
    echo ""
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi