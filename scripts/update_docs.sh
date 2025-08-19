#!/bin/bash
# update_docs.sh - PillSnap ML 문서 자동 업데이트 스크립트
# 사용법: bash scripts/update_docs.sh

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 확인
PROJECT_ROOT="/home/max16/pillsnap"
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo -e "${RED}❌ 프로젝트 루트를 찾을 수 없습니다: $PROJECT_ROOT${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 PillSnap ML 문서 자동 업데이트 시작${NC}"
echo "=================================================="

# 현재 상황 정보 수집
echo -e "${YELLOW}📋 현재 상황 정보 수집 중...${NC}"

# 디스크 사용량 확인
SSD_USAGE=$(du -sh /home/max16/ssd_pillsnap/dataset 2>/dev/null | cut -f1 || echo "N/A")
SSD_FILES=$(find /home/max16/ssd_pillsnap/dataset -name "*.png" 2>/dev/null | wc -l || echo "0")

# 프로젝트 파일 수 확인
PYTHON_FILES=$(find src/ -name "*.py" | wc -l)
TEST_FILES=$(find tests/ -name "*.py" | wc -l)

# 현재 날짜
CURRENT_DATE=$(date '+%Y-%m-%d')

echo "📊 현재 상태:"
echo "  - SSD 데이터: $SSD_USAGE ($SSD_FILES 파일)"
echo "  - Python 파일: $PYTHON_FILES 개"
echo "  - 테스트 파일: $TEST_FILES 개"
echo "  - 업데이트 날짜: $CURRENT_DATE"
echo ""

# 업데이트 대상 문서 목록
DOCS_TO_UPDATE=(
    ".claude/commands/initial-prompt.md"
    "CLAUDE.md"
    "README.md"
    "Prompt/PART_0.md"
    "Prompt/PART_A.md"
    "Prompt/PART_B.md"
    "Prompt/PART_C.md"
    "Prompt/PART_D.md"
    "Prompt/PART_E.md"
    "Prompt/PART_F.md"
    "Prompt/PART_G.md"
    "Prompt/PART_H.md"
)

# 백업 디렉토리 생성
BACKUP_DIR="artifacts/backups/docs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo -e "${YELLOW}💾 문서 백업 중...${NC}"
for doc in "${DOCS_TO_UPDATE[@]}"; do
    if [[ -f "$doc" ]]; then
        cp "$doc" "$BACKUP_DIR/"
        echo "  ✅ $doc 백업 완료"
    else
        echo -e "  ${RED}⚠️ $doc 파일을 찾을 수 없습니다${NC}"
    fi
done

# Claude Code 업데이트 명령어 실행 안내
echo ""
echo -e "${GREEN}🤖 Claude Code 문서 업데이트 실행${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}다음 명령어를 Claude Code에서 실행하세요:${NC}"
echo ""
echo -e "${YELLOW}/.claude/commands/update-doc.md${NC}"
echo ""
echo "이 명령어가 다음 작업을 수행합니다:"
echo "  1. 모든 PART_*.md 파일 현재 상황 반영"
echo "  2. initial-prompt.md 업데이트"
echo "  3. README.md 업데이트"
echo "  4. 경로 정보 일괄 변경 (HDD → SSD)"
echo "  5. M.2 SSD 확장 계획 반영"
echo ""

# 검증 스크립트 제공
echo -e "${YELLOW}🔍 업데이트 후 검증 명령어:${NC}"
echo ""
echo "# SSD 데이터 확인"
echo "ls -la /home/max16/ssd_pillsnap/dataset/"
echo "du -sh /home/max16/ssd_pillsnap/dataset/"
echo ""
echo "# 설정 파일 확인"
echo "grep -n 'ssd_pillsnap' config.yaml"
echo ""
echo "# 프로젝트 구조 확인"
echo "tree -L 3 src/"
echo ""

echo -e "${GREEN}✅ 문서 업데이트 준비 완료${NC}"
echo "백업 위치: $BACKUP_DIR"
echo ""
echo -e "${BLUE}📝 다음 단계:${NC}"
echo "1. Claude Code에서 '/.claude/commands/update-doc.md' 실행"
echo "2. 업데이트 완료 후 검증 명령어 실행"
echo "3. git add . && git commit -m 'docs: 디스크 I/O 병목 해결 문서 업데이트'"
echo ""
echo "=================================================="
echo -e "${GREEN}🚀 Ready for Document Update!${NC}"