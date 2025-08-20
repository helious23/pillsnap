#!/bin/bash
# Stage 2 데이터 이전 실시간 모니터링 스크립트

echo "🔍 Stage 2 데이터 이전 모니터링"
echo "================================="
echo ""

# 프로세스 확인
echo "📊 실행 중인 프로세스:"
ps aux | grep "migrate_stage2_data" | grep -v grep
echo ""

# SSD 용량 확인
echo "💾 SSD 사용량:"
df -h /home/max16/ssd_pillsnap | tail -1
echo ""

# 현재 이전된 클래스 수 확인
echo "📁 현재 SSD에 있는 K-코드 수:"
find /home/max16/ssd_pillsnap/dataset/data/train/images/single -type d -name "K-*" | wc -l
echo ""

# 최근 로그 확인 (로그 파일이 있다면)
echo "📝 최근 활동 (파일 생성 시간 기준):"
find /home/max16/ssd_pillsnap/dataset/data/train/images/single -name "*.png" -newermt "10 minutes ago" | head -5
echo ""

# 실시간 모니터링 시작
echo "🔄 실시간 모니터링 (Ctrl+C로 종료):"
echo "새로 생성되는 파일들을 실시간으로 표시합니다..."
echo ""

# inotify로 실시간 파일 생성 모니터링
if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -m -r --format '%T %w %f' --timefmt '%H:%M:%S' \
        -e create /home/max16/ssd_pillsnap/dataset/data/train/images/single/ 2>/dev/null | \
        while read time dir file; do
            if [[ "$file" == *.png ]]; then
                echo "[$time] 복사 완료: $file"
            fi
        done
else
    echo "⚠️  inotifywait가 설치되지 않아 실시간 모니터링을 사용할 수 없습니다."
    echo "대신 주기적으로 상태를 확인합니다..."
    
    while true; do
        echo "[$(date '+%H:%M:%S')] 현재 K-코드 수: $(find /home/max16/ssd_pillsnap/dataset/data/train/images/single -type d -name "K-*" | wc -l)"
        sleep 30
    done
fi