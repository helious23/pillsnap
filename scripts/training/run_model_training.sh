#!/bin/bash
# Original PART_B Design Training Script
# Two-Stage Interleaved Training with OptimizationAdvisor

set -euo pipefail

VENV="$HOME/pillsnap/.venv"
ROOT="/home/max16/pillsnap"

source "$VENV/bin/activate" && cd "$ROOT"

# 실험 디렉토리 생성 보장
EXP_DIR=$(yq '.paths.exp_dir' config.yaml)
mkdir -p "$EXP_DIR"/{logs,tb,reports,checkpoints,export}

# 설정 요약 출력
echo "🚀 Two-Stage Conditional Pipeline Training"
echo "   Strategy: $(yq '.train.strategy' config.yaml)"
echo "   Current Stage: $(yq '.progressive_validation.current_stage' config.yaml)"
echo "   AMP: $(yq '.optimization.amp' config.yaml)"
echo "   Compile: $(yq '.optimization.torch_compile' config.yaml)"

# TODO: PART_D에서 실제 구현
echo "⚠️  Training implementation pending PART_D"
echo "    Current: Using existing pillsnap.stage2.train_cls"
echo "    Target: src.train with Two-Stage Pipeline"

# 임시로 현재 구현 호출 (PART_D까지의 브릿지)
/home/max16/pillsnap/.venv_gpu/bin/python -m pillsnap.stage2.train_cls \
  --manifest artifacts/manifest_enriched.csv \
  --classes artifacts/classes_step11.json \
  --device cuda \
  --epochs 1 \
  --batch-size 4 \
  --limit 16 \
  --outdir "$EXP_DIR/checkpoints/bridge_run_$(date +%s)"

echo "✅ Bridge training completed. Proceed with PART_C+D implementation."
