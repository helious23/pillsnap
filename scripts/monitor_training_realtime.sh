#!/bin/bash
"""
Stage 3 훈련과 실시간 모니터링을 동시에 실행하는 스크립트

사용법:
  ./scripts/monitor_training_realtime.sh [훈련_스크립트] [모니터링_포트]

예제:
  ./scripts/monitor_training_realtime.sh "python -m src.train --cfg config.yaml" 8888
  ./scripts/monitor_training_realtime.sh "./scripts/train_stage3.sh" 9999
"""

set -e

# 기본값 설정
TRAINING_COMMAND=${1:-"echo 'Stage 3 훈련 시뮬레이션...'; sleep 3600"}
MONITOR_PORT=${2:-8888}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🚀 Stage 3 훈련 + 실시간 모니터링 시작"
echo "=" * 50
echo "📂 프로젝트 루트: $PROJECT_ROOT"
echo "🏃 훈련 명령어: $TRAINING_COMMAND"
echo "🌐 모니터링 포트: $MONITOR_PORT"
echo "📊 대시보드: http://localhost:$MONITOR_PORT"
echo "=" * 50

# 가상환경 활성화
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "🐍 가상환경 활성화..."
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "⚠️  가상환경을 찾을 수 없습니다. 시스템 Python 사용"
fi

# 로그 디렉토리 생성
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TRAINING_LOG="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo "📋 훈련 로그: $TRAINING_LOG"

# 종료 시그널 처리
cleanup() {
    echo ""
    echo "🛑 종료 신호 감지. 정리 중..."
    
    # 백그라운드 프로세스들 종료
    if [ ! -z "$TRAINING_PID" ]; then
        echo "📚 훈련 프로세스 종료 중... (PID: $TRAINING_PID)"
        kill -TERM $TRAINING_PID 2>/dev/null || true
        wait $TRAINING_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$MONITOR_PID" ]; then
        echo "📊 모니터링 서버 종료 중... (PID: $MONITOR_PID)"
        kill -TERM $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
    fi
    
    echo "✅ 정리 완료"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 1. 모니터링 서버 백그라운드 시작
echo "📊 실시간 모니터링 서버 시작..."
cd "$PROJECT_ROOT"
python scripts/start_stage3_monitor.py \
    --port "$MONITOR_PORT" \
    --log-cmd "$TRAINING_COMMAND" &
MONITOR_PID=$!

# 모니터링 서버가 시작될 때까지 대기
echo "⏳ 모니터링 서버 초기화 대기..."
sleep 3

# 모니터링 서버 상태 확인
if ! kill -0 $MONITOR_PID 2>/dev/null; then
    echo "❌ 모니터링 서버 시작 실패"
    exit 1
fi

echo "✅ 모니터링 서버 시작됨 (PID: $MONITOR_PID)"
echo "🌐 대시보드: http://localhost:$MONITOR_PORT"
echo ""

# 2. 훈련 명령어 실행 (백그라운드)
echo "🏃 훈련 시작..."
echo "📋 로그 파일: $TRAINING_LOG"
echo ""

# 훈련 명령어를 로그 파일에 출력하면서 실행
(
    echo "=== Stage 3 훈련 시작: $(date) ==="
    echo "명령어: $TRAINING_COMMAND"
    echo "==================================="
    echo ""
    
    # 실제 훈련 실행
    eval "$TRAINING_COMMAND" 2>&1
    
    echo ""
    echo "=== 훈련 완료: $(date) ==="
) | tee "$TRAINING_LOG" &

TRAINING_PID=$!
echo "🏃 훈련 프로세스 시작됨 (PID: $TRAINING_PID)"

# 3. 상태 출력
echo ""
echo "📋 실시간 모니터링 대시보드가 실행 중입니다!"
echo ""
echo "🌐 웹 브라우저에서 다음 주소로 접속하세요:"
echo "   http://localhost:$MONITOR_PORT"
echo ""
echo "📊 실시간으로 다음 정보를 확인할 수 있습니다:"
echo "   • 훈련 로그 스트리밍"
echo "   • GPU/CPU 사용률"
echo "   • Progressive Resize 진행상황"  
echo "   • Stage 4 진입 준비도"
echo "   • 최적화 권고사항"
echo ""
echo "⏹️  종료하려면 Ctrl+C를 누르세요"
echo ""

# 4. 훈련 완료까지 대기
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "🎉 훈련이 성공적으로 완료되었습니다!"
else
    echo "❌ 훈련이 오류로 종료되었습니다 (Exit code: $TRAINING_EXIT_CODE)"
fi

echo "📋 훈련 로그: $TRAINING_LOG"
echo "📊 모니터링은 계속 실행 중입니다. Ctrl+C로 종료하세요."

# 5. 모니터링만 계속 실행
echo ""
echo "⏳ 모니터링 서버는 계속 실행됩니다..."
echo "   대시보드: http://localhost:$MONITOR_PORT"
echo "   종료: Ctrl+C"

wait $MONITOR_PID