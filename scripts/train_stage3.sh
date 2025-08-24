#!/bin/bash
"""
Stage 3 Classification 전용 학습 실행 스크립트 (단순 래퍼)

로그 통일을 위해 Python 직접 실행만 수행
모든 로그는 src.training.train_stage3_classification_*.log로 통일됨
"""

set -e

# 프로젝트 루트 경로
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Python 가상환경 활성화
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "✅ Python 가상환경 활성화됨"
else
    echo "❌ Python 가상환경을 찾을 수 없습니다: .venv/bin/activate"
    exit 1
fi

echo "🚀 Stage 3 Classification 학습 시작"
echo "📝 모든 로그는 src.training.train_stage3_classification_*.log 파일에서 확인 가능"
echo ""

# Python 학습 스크립트 직접 실행 (모든 인수 그대로 전달)
python3 -m src.training.train_stage3_classification "$@"

echo ""
echo "✅ 학습 완료!"
echo "📊 결과 확인: /home/max16/pillsnap_data/exp/exp01/logs/src.training.train_stage3_classification_$(date +%Y%m%d).log"