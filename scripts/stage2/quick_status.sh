#!/bin/bash
# Stage 2 데이터 이전 빠른 상태 확인

echo "📊 Stage 2 데이터 이전 현황"
echo "========================="

# 프로세스 상태 (더 간단하고 확실한 검사)
if pgrep -f "migrate_stage2_data.py" >/dev/null 2>&1; then
    echo "✅ 이전 프로세스: 실행 중"
    # 실행 시간도 표시
    RUNTIME=$(ps -o etime= -p $(pgrep -f "migrate_stage2_data.py") 2>/dev/null | tr -d ' ')
    if [ -n "$RUNTIME" ]; then
        echo "⏱️  실행 시간: $RUNTIME"
    fi
else
    echo "❌ 이전 프로세스: 중지됨"
fi

# 현재 클래스 수
CURRENT_CLASSES=$(find /home/max16/ssd_pillsnap/dataset/data/train/images/single -type d -name "K-*" | wc -l)
echo "📁 현재 K-코드 수: $CURRENT_CLASSES개"

# 목표 대비 진행률 (Stage 1: 51개 → Stage 2: 51 + 237 = 288개)
TARGET_CLASSES=288
PROGRESS=$(echo "scale=1; ($CURRENT_CLASSES * 100) / $TARGET_CLASSES" | bc -l 2>/dev/null || echo "계산불가")
echo "📈 진행률: $PROGRESS% ($CURRENT_CLASSES/$TARGET_CLASSES)"

# SSD 사용량
echo "💾 SSD 사용량:"
df -h /home/max16/ssd_pillsnap | tail -1

# 최근 5분간 생성된 파일 수 (생성 시간 기준)
RECENT_FILES=$(find /home/max16/ssd_pillsnap/dataset/data/train/images/single -name "*.png" -cmin -5 | wc -l)
echo "🆕 최근 5분간 복사된 이미지: $RECENT_FILES개"

# Stage 1에서 증가한 클래스 수 계산 (Stage 1: 51개 → 현재)
STAGE1_CLASSES=51
NEW_CLASSES=$((CURRENT_CLASSES - STAGE1_CLASSES))
echo "📈 Stage 2 신규 추가: $NEW_CLASSES개 클래스"

echo ""
echo "🔄 실시간 모니터링: ./scripts/stage2/monitor_stage2_migration.sh"
echo "📊 상태 재확인: ./scripts/stage2/quick_status.sh"