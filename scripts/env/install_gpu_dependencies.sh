#!/bin/bash
# GPU 스모크 테스트에 필요한 최소 패키지만 설치

set -euo pipefail
cd /home/max16/pillsnap
source .venv_gpu/bin/activate

echo "🔧 Installing minimal packages for GPU smoke tests..."

# Stage2에 필요한 최소 패키지들
pip install \
  pyyaml \
  tqdm \
  pandas \
  scikit-learn \
  rich

echo "✅ Minimal packages installed for GPU testing"
echo "ℹ️ Additional packages will be installed as needed:"
echo "  - timm (for EfficientNet)"  
echo "  - ultralytics (for YOLO)"
echo "  - opencv (for image processing)"
echo "  - fastapi (for API server)"