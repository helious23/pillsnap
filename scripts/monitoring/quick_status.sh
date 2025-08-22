#!/bin/bash
# PillSnap Quick Status Check
# 현재 학습 상태를 간단히 확인하는 스크립트

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 PillSnap 빠른 상태 확인${NC}"
echo "================================="

# 1. 프로세스 확인
echo -n "📊 학습 프로세스: "
if pgrep -f "train_classification_stage" > /dev/null; then
    echo -e "${GREEN}실행 중 ✅${NC}"
    ps aux | grep "train_classification_stage" | grep -v grep | awk '{printf "PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
else
    echo -e "${RED}없음 ❌${NC}"
fi

echo

# 2. GPU 확인
echo -n "🎮 GPU 상태: "
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}사용 가능 ✅${NC}"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | while IFS=',' read -r name util mem_used mem_total; do
        name=$(echo "$name" | xargs)
        util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        mem_percent=$((mem_used * 100 / mem_total))
        printf "  %s: %s%% 사용률, %s/%sMB (%s%% 메모리)\n" "$name" "$util" "$mem_used" "$mem_total" "$mem_percent"
    done
else
    echo -e "${RED}nvidia-smi 없음 ❌${NC}"
fi

echo

# 3. 최근 모델 확인
echo "🎯 완료된 Stage:"
models_dir="artifacts/models/classification"
if [ -d "$models_dir" ]; then
    for stage in 1 2 3 4; do
        case $stage in
            1) classes=50 ;;
            2) classes=250 ;;
            3) classes=1000 ;;
            4) classes=4523 ;;
        esac
        
        model_file="$models_dir/best_classifier_${classes}classes.pt"
        if [ -f "$model_file" ]; then
            echo -e "  Stage $stage: ${GREEN}완료 ✅${NC} (${classes} 클래스)"
            # 정확도 확인 시도
            accuracy=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$model_file', map_location='cpu')
    acc = checkpoint.get('best_accuracy', 0)
    print(f'    정확도: {acc:.1%}')
except:
    print('    정확도: N/A')
" 2>/dev/null || echo "    정확도: N/A")
            echo "$accuracy"
        else
            echo -e "  Stage $stage: ${YELLOW}미완료 ⏳${NC}"
        fi
    done
else
    echo -e "  ${RED}모델 디렉토리 없음${NC}"
fi

echo

# 4. 디스크 공간 확인
echo "💾 디스크 공간:"
df -h . | tail -1 | awk '{printf "  사용률: %s (%s 사용됨, %s 사용 가능)\n", $5, $3, $4}'

echo
echo -e "${BLUE}상세 모니터링: ./scripts/monitoring/universal_training_monitor.sh${NC}"