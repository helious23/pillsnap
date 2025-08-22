#!/bin/bash
# PillSnap Monitoring Aliases Setup
# 모니터링 스크립트를 위한 편리한 별칭 설정

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat << 'EOF'
# PillSnap Monitoring Aliases
# ~/.bashrc 또는 ~/.zshrc에 다음 라인들을 추가하세요:

# 통합 모니터링 (추천)
alias monitor="$SCRIPT_DIR/universal_training_monitor.sh"
alias mon="$SCRIPT_DIR/universal_training_monitor.sh"

# 빠른 상태 확인
alias status="$SCRIPT_DIR/quick_status.sh" 
alias st="$SCRIPT_DIR/quick_status.sh"

# Stage별 모니터링
alias mon1="$SCRIPT_DIR/universal_training_monitor.sh --stage 1"
alias mon2="$SCRIPT_DIR/universal_training_monitor.sh --stage 2"
alias mon3="$SCRIPT_DIR/universal_training_monitor.sh --stage 3"
alias mon4="$SCRIPT_DIR/universal_training_monitor.sh --stage 4"

# 고빈도 모니터링
alias monfast="$SCRIPT_DIR/universal_training_monitor.sh --interval 1"

# GPU 정보만
alias gpu="nvidia-smi"
alias gpuw="watch -n 1 nvidia-smi"

EOF

echo
echo "위 내용을 ~/.bashrc에 추가하려면:"
echo "echo '# PillSnap Monitoring Aliases' >> ~/.bashrc"
echo "echo 'source $SCRIPT_DIR/setup_aliases.sh 2>/dev/null || true' >> ~/.bashrc"
echo "source ~/.bashrc"
echo
echo "사용 예시:"
echo "  monitor    # 통합 모니터링"
echo "  status     # 빠른 상태 확인"
echo "  mon2       # Stage 2 모니터링"
echo "  monfast    # 1초마다 새로고침"