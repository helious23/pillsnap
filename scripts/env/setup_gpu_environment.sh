#!/bin/bash
# RTX 5080 sm_120 호환 PyTorch 2.7.0+cu128 환경 설정
# 현재 패키지 버전 기준으로 호환성 유지

set -euo pipefail

VENV="$HOME/pillsnap/.venv"
ROOT="/home/max16/pillsnap"

echo "🔧 Setting up RTX 5080 compatible PyTorch environment..."

# 1. 기존 venv 활성화
source "$VENV/bin/activate" && cd "$ROOT"

# 2. PyTorch 2.7.0+cu128 설치 (RTX 5080 sm_120 지원)
echo "📦 Installing PyTorch 2.7.0+cu128 for RTX 5080 sm_120..."

# 기존 PyTorch 제거
pip uninstall -y torch torchvision torchaudio || true

# RTX 5080 호환 PyTorch 설치 (CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "📦 Installing additional packages to match requirements..."

# 현재 requirements.txt의 다른 패키지들 설치
pip install timm==1.0.19
pip install ultralytics==8.3.179
pip install "numpy>=2.1.0,<2.2"
pip install "pillow>=10.0,<12.0"
pip install "opencv-python-headless>=4.9,<5.0"
pip install "albumentations>=1.4,<2.0"
pip install "kornia>=0.7,<1.0"
pip install "pyyaml>=6.0,<7.0"
pip install "tqdm>=4.66,<5.0"
pip install "pandas>=2.0,<3.0"
pip install "scikit-learn>=1.4,<2.0"
pip install "tensorboard>=2.15,<3.0"
pip install "matplotlib>=3.8,<4.0"
pip install "seaborn>=0.13,<1.0"
pip install "fastapi>=0.110,<1.0"
pip install "uvicorn[standard]>=0.27,<1.0"
pip install "python-multipart>=0.0.9,<1.0"
pip install "pydantic>=2.6,<3.0"
pip install "pydantic-settings>=2.2,<3.0"
pip install "python-dotenv>=1.0,<2.0"
pip install "onnx>=1.17.0"
pip install "onnxruntime-gpu>=1.20.0"
pip install "lmdb>=1.4,<2.0"
pip install "psutil>=5.9,<6.0"
pip install "httpx>=0.26,<1.0"
pip install "pytest>=8.0,<9.0"
pip install "pytest-asyncio>=0.23,<1.0"

echo "🔍 Verifying installation..."

# 3. RTX 5080 호환성 검증
python - <<'PY'
import torch
import sys

print("=== PyTorch Installation Verification ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm_{capability[0]}{capability[1]}")
    
    # RTX 5080 sm_120 테스트
    if capability[0] >= 12:  # sm_120
        print("✅ RTX 5080 sm_120 compatibility confirmed!")
        
        # 간단한 GPU 연산 테스트
        x = torch.randn(100, 100, device='cuda')
        y = torch.mm(x, x.T)
        print(f"✅ GPU computation test passed: {y.shape}")
        
        # Mixed Precision 테스트
        with torch.amp.autocast('cuda'):
            z = torch.mm(x, x.T)
        print(f"✅ AMP test passed: {z.shape}")
        
    else:
        print(f"⚠️  Compute capability sm_{capability[0]}{capability[1]} < sm_120")
else:
    print("❌ CUDA not available")
    sys.exit(1)

# 4. 기타 패키지 호환성 검증
try:
    import timm
    import ultralytics
    import numpy as np
    print(f"✅ TIMM: {timm.__version__}")
    print(f"✅ Ultralytics: {ultralytics.__version__}")
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ Package import error: {e}")
    sys.exit(1)

print("\n🎉 RTX 5080 environment setup completed successfully!")
PY

echo "✅ RTX 5080 compatible environment ready!"
echo ""
echo "📋 Next steps:"
echo "   1. Test GPU with: python tests/test_progressive_validation.py"
echo "   2. Run Stage 1 evaluation: python tests/stage_1_evaluator.py"
echo "   3. Begin PART_C implementation: src/data.py"